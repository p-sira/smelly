use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use nalgebra::DMatrix;
use smelly::ParticleFilter;

struct CountingAllocator;
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

fn record_allocation() {
    // Ignore allocations on other threads and during thread-local teardown.
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(current) = count.get() {
            count.set(Some(current + 1));
        }
    });
}

// Forward allocations unchanged, counting only the measured filter operations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
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
    ALLOCATIONS.set(Some(0));
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
    assert_eq!(ALLOCATIONS.replace(None), Some(0));
}

#[test]
fn allocation_tracking_counts_only_the_current_thread() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let start = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            while !start.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            std::hint::black_box(Box::new(42));
            done.store(true, Ordering::Release);
        });
        ALLOCATIONS.set(Some(0));
        start.store(true, Ordering::Release);
        while !done.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        std::hint::black_box(Box::new(42));
        let allocations = ALLOCATIONS.replace(None);
        worker.join().unwrap();
        assert_eq!(allocations, Some(1));
    });
}
