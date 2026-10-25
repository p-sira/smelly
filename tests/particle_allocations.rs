use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nalgebra::DMatrix;
use smelly::ParticleFilter;

struct CountingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// Forward allocations unchanged, counting only the measured filter operations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn predict_update_and_resample_allocate_nothing_after_construction() {
    let mut filter = ParticleFilter::<f64>::new(DMatrix::from_fn(4, 1024, |row, i| {
        (i as f64 * 0.1 + row as f64).sin()
    }))
    .unwrap();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    for _ in 0..10 {
        filter.predict(|x, out| out[0] = x[0] + 0.1 * x[2]).unwrap();
        filter.update(|x| (-0.5 * x[0] * x[0]).exp()).unwrap();
        filter.resample(0.37).unwrap();
    }
    // Static storage also avoids allocations during construction and moments.
    {
        use nalgebra::{SMatrix, SVector};
        use smelly::{
            MerweScaledSigmaPoints, SExtendedKalmanFilter, SKalmanFilter, SParticleFilter,
            SUnscentedKalmanFilter,
        };
        let mut linear = SKalmanFilter::<f64, 4, 2>::new();
        let mut extended = SExtendedKalmanFilter::<f64, 4, 2>::new();
        let mut unscented =
            SUnscentedKalmanFilter::<f64, 4, 2>::new(MerweScaledSigmaPoints::new(0.1, 2.0, 0.0));
        let h = SMatrix::<f64, 2, 4>::identity();
        linear.set_measurement(h).unwrap();
        let z = SVector::<f64, 2>::repeat(1.0);
        let mut particles = SParticleFilter::<f64, 4, 128>::new(SMatrix::from_fn(|row, i| {
            (i as f64 * 0.1 + row as f64).sin()
        }))
        .unwrap();
        for _ in 0..10 {
            linear.predict().unwrap();
            std::hint::black_box(linear.update(&z).unwrap());
            extended.predict().unwrap();
            std::hint::black_box(extended.update(&z, |x| h * x, |_| h).unwrap());
            unscented.predict(|x| *x).unwrap();
            std::hint::black_box(unscented.update(&z, |x| h * x).unwrap());
            particles
                .predict(|x, out| out[0] = x[0] + 0.1 * x[2])
                .unwrap();
            particles.update(|x| (-0.5 * x[0] * x[0]).exp()).unwrap();
            particles
                .update_log_likelihood(|x| -0.5 * x[0] * x[0])
                .unwrap();
            particles.resample(0.37).unwrap();
            std::hint::black_box(particles.state());
            std::hint::black_box(particles.covariance());
        }
    }
    TRACK.store(false, Ordering::Relaxed);
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}
