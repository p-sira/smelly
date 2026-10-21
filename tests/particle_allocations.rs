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
    TRACK.store(false, Ordering::Relaxed);
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}
