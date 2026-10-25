//! Compare identical initial samples, motion, and likelihoods for each cycle.
//! Setup and destruction are excluded with `iter_batched_ref`. Process noise
//! and automatic resampling are disabled; the second group explicitly performs
//! one systematic resampling. Smelly receives a fixed offset, while the reference
//! draws its offset internally because its public API does not accept one.

use core::hint::black_box;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use kalman_filters::ParticleFilterBuilder;
use nalgebra::DMatrix;
use smelly::ParticleFilter;

fn likelihood(state: &[f64]) -> f64 {
    (-0.5 * (state[0] * state[0] + state[1] * state[1])).exp()
}

fn compare(c: &mut Criterion) {
    for resample in [false, true] {
        let mut group = c.benchmark_group(if resample {
            "particle/predict_update_resample"
        } else {
            "particle/predict_update"
        });
        for count in [128, 1024, 8192] {
            let samples = DMatrix::from_fn(4, count, |row, i| (i as f64 * 0.1 + row as f64).sin());
            let filter = ParticleFilter::new(samples.clone()).unwrap();
            group.bench_with_input(BenchmarkId::new("smelly", count), &count, |b, _| {
                b.iter_batched_ref(
                    || filter.clone(),
                    |filter| {
                        filter
                            .predict(|x, out| {
                                out[0] = x[0] + 0.1 * x[2];
                                out[1] = x[1] + 0.1 * x[3];
                            })
                            .unwrap();
                        filter.update(|x| likelihood(black_box(x))).unwrap();
                        if resample {
                            filter.resample(black_box(0.37)).unwrap();
                        }
                        black_box(filter);
                    },
                    BatchSize::LargeInput,
                );
            });
            group.bench_with_input(
                BenchmarkId::new("kalman-filter-rs", count),
                &count,
                |b, _| {
                    b.iter_batched_ref(
                        || {
                            let mut filter = ParticleFilterBuilder::new(4, count)
                                .initial_mean(vec![0.0; 4])
                                .initial_std(vec![0.0; 4])
                                .process_noise_std(vec![0.0; 4])
                                .measurement_noise_std(vec![1.0; 2])
                                .dt(0.1)
                                .ess_threshold(0.0)
                                .build()
                                .unwrap();
                            for (particle, column) in
                                filter.particles.iter_mut().zip(samples.column_iter())
                            {
                                particle.state.copy_from_slice(column.as_slice());
                            }
                            filter
                        },
                        |filter| {
                            filter.predict(|x, dt| {
                                vec![x[0] + dt * x[2], x[1] + dt * x[3], x[2], x[3]]
                            });
                            filter
                                .update(&[0.0, 0.0], |x, _| likelihood(black_box(x)))
                                .unwrap();
                            if resample {
                                filter.resample().unwrap();
                            }
                            black_box(filter);
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
        group.finish();
    }
}

fn updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle/update");
    let mut filter = ParticleFilter::<f64>::new(DMatrix::from_fn(4, 1024, |row, i| {
        (i as f64 * 0.1 + row as f64).sin()
    }))
    .unwrap();
    filter.update(likelihood).unwrap();
    for (name, scale) in [
        ("linear", 1.0),
        ("tiny_linear", f64::MIN_POSITIVE),
        ("log", 1.0),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                || filter.clone(),
                |filter| {
                    if name == "log" {
                        filter
                            .update_log_likelihood(|x| {
                                let x = black_box(x);
                                -0.5 * (x[0] * x[0] + x[1] * x[1])
                            })
                            .unwrap();
                    } else {
                        filter.update(|x| scale * likelihood(black_box(x))).unwrap();
                    }
                    black_box(filter);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, compare, updates);
criterion_main!(benches);
