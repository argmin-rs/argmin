use argmin_testfunctions::{
    ackley, ackley_derivative, ackley_derivative_const, ackley_hessian, ackley_hessian_const,
    beale, beale_derivative, beale_hessian, booth, booth_derivative, booth_hessian, bukin_n6,
    bukin_n6_derivative, bukin_n6_hessian, cross_in_tray, cross_in_tray_derivative,
    cross_in_tray_hessian, easom, easom_derivative, easom_hessian, eggholder, eggholder_derivative,
    eggholder_hessian, goldsteinprice, goldsteinprice_derivative, goldsteinprice_hessian,
    himmelblau, himmelblau_derivative, himmelblau_hessian, holder_table, holder_table_derivative,
    holder_table_hessian, levy, levy_derivative, levy_derivative_const, levy_hessian,
    levy_hessian_const, levy_n13, levy_n13_derivative, levy_n13_hessian, matyas, matyas_derivative,
    matyas_hessian, mccorminck, mccorminck_derivative, mccorminck_hessian, picheny,
    picheny_derivative, picheny_hessian, rastrigin, rastrigin_derivative,
    rastrigin_derivative_const, rastrigin_hessian, rastrigin_hessian_const, rosenbrock,
    rosenbrock_derivative, rosenbrock_derivative_const, rosenbrock_hessian,
    rosenbrock_hessian_const, schaffer_n2, schaffer_n2_derivative, schaffer_n2_hessian,
    schaffer_n4, schaffer_n4_derivative, schaffer_n4_hessian, sphere, sphere_derivative,
    sphere_derivative_const, sphere_hessian, sphere_hessian_const, styblinski_tang,
    styblinski_tang_derivative, styblinski_tang_derivative_const, styblinski_tang_hessian,
    styblinski_tang_hessian_const, threehumpcamel, threehumpcamel_derivative,
    threehumpcamel_hessian, zero, zero_derivative, zero_derivative_const, zero_hessian,
    zero_hessian_const,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const P2: &[f64; 2] = &[1.0; 2];
const P10: &[f64; 10] = &[1.0; 10];
const P20: &[f64; 20] = &[1.0; 20];

pub fn bm_ackley(c: &mut Criterion) {
    let mut g = c.benchmark_group("ackley");
    // Test function
    g.bench_function("ackley 02", |b| b.iter(|| ackley(black_box(P2))));
    g.bench_function("ackley 10", |b| b.iter(|| ackley(black_box(P10))));
    g.bench_function("ackley 20", |b| b.iter(|| ackley(black_box(P20))));
    // Derivative
    g.bench_function("ackley_derivative 02", |b| {
        b.iter(|| ackley_derivative(black_box(P2)))
    });
    g.bench_function("ackley_derivative 10", |b| {
        b.iter(|| ackley_derivative(black_box(P10)))
    });
    g.bench_function("ackley_derivative 20", |b| {
        b.iter(|| ackley_derivative(black_box(P20)))
    });
    // Derivative const
    g.bench_function("ackley_derivative_const 02", |b| {
        b.iter(|| ackley_derivative_const(black_box(P2)))
    });
    g.bench_function("ackley_derivative_const 10", |b| {
        b.iter(|| ackley_derivative_const(black_box(P10)))
    });
    g.bench_function("ackley_derivative_const 20", |b| {
        b.iter(|| ackley_derivative_const(black_box(P20)))
    });
    // Hessian
    g.bench_function("ackley_hessian 02", |b| {
        b.iter(|| ackley_hessian(black_box(P2)))
    });
    g.bench_function("ackley_hessian 10", |b| {
        b.iter(|| ackley_hessian(black_box(P10)))
    });
    g.bench_function("ackley_hessian 20", |b| {
        b.iter(|| ackley_hessian(black_box(P20)))
    });
    // Hessian cons
    g.bench_function("ackley_hessian_const 02", |b| {
        b.iter(|| ackley_hessian_const(black_box(P2)))
    });
    g.bench_function("ackley_hessian_const 10", |b| {
        b.iter(|| ackley_hessian_const(black_box(P10)))
    });
    g.bench_function("ackley_hessian_const 20", |b| {
        b.iter(|| ackley_hessian_const(black_box(P20)))
    });
    g.finish();
}

pub fn bm_beale(c: &mut Criterion) {
    let mut g = c.benchmark_group("beale");
    // Test function
    g.bench_function("beale 2", |b| b.iter(|| beale(black_box(P2))));
    // Derivative
    g.bench_function("beale_derivative 2", |b| {
        b.iter(|| beale_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("beale_hessian 2", |b| {
        b.iter(|| beale_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_booth(c: &mut Criterion) {
    let mut g = c.benchmark_group("booth");
    // Test function
    g.bench_function("booth 2", |b| b.iter(|| booth(black_box(P2))));
    // Derivative
    g.bench_function("booth_derivative 2", |b| {
        b.iter(|| booth_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("booth_hessian 2", |b| {
        b.iter(|| booth_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_bukin_n6(c: &mut Criterion) {
    let mut g = c.benchmark_group("bukin_n6");
    // Test function
    g.bench_function("bukin_n6 2", |b| b.iter(|| bukin_n6(black_box(P2))));
    // Derivative
    g.bench_function("bukin_n6_derivative 2", |b| {
        b.iter(|| bukin_n6_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("bukin_n6_hessian 2", |b| {
        b.iter(|| bukin_n6_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_cross_in_tray(c: &mut Criterion) {
    let mut g = c.benchmark_group("cross_in_tray");
    // Test function
    g.bench_function("cross_in_tray 2", |b| {
        b.iter(|| cross_in_tray(black_box(P2)))
    });
    // Derivative
    g.bench_function("cross_in_tray_derivative 2", |b| {
        b.iter(|| cross_in_tray_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("cross_in_tray_hessian 2", |b| {
        b.iter(|| cross_in_tray_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_easom(c: &mut Criterion) {
    let mut g = c.benchmark_group("easom");
    // Test function
    g.bench_function("easom 2", |b| b.iter(|| easom(black_box(P2))));
    // Derivative
    g.bench_function("easom_derivative 2", |b| {
        b.iter(|| easom_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("easom_hessian 2", |b| {
        b.iter(|| easom_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_eggholder(c: &mut Criterion) {
    let mut g = c.benchmark_group("eggholder");
    // Test function
    g.bench_function("eggholder 2", |b| b.iter(|| eggholder(black_box(P2))));
    // Derivative
    g.bench_function("eggholder_derivative 2", |b| {
        b.iter(|| eggholder_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("eggholder_hessian 2", |b| {
        b.iter(|| eggholder_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_goldsteinprice(c: &mut Criterion) {
    let mut g = c.benchmark_group("goldsteinprice");
    // Test function
    g.bench_function("goldsteinprice 2", |b| {
        b.iter(|| goldsteinprice(black_box(P2)))
    });
    // Derivative
    g.bench_function("goldsteinprice_derivative 2", |b| {
        b.iter(|| goldsteinprice_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("goldsteinprice_hessian 2", |b| {
        b.iter(|| goldsteinprice_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_himmelblau(c: &mut Criterion) {
    let mut g = c.benchmark_group("himmelblau");
    // Test function
    g.bench_function("himmelblau 2", |b| b.iter(|| himmelblau(black_box(P2))));
    // Derivative
    g.bench_function("himmelblau_derivative 2", |b| {
        b.iter(|| himmelblau_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("himmelblau_hessian 2", |b| {
        b.iter(|| himmelblau_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_holder_table(c: &mut Criterion) {
    let mut g = c.benchmark_group("holder_table");
    // Test function
    g.bench_function("holder_table 2", |b| b.iter(|| holder_table(black_box(P2))));
    // Derivative
    g.bench_function("holder_table_derivative 2", |b| {
        b.iter(|| holder_table_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("holder_table_hessian 2", |b| {
        b.iter(|| holder_table_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_levy(c: &mut Criterion) {
    let mut g = c.benchmark_group("levy");
    // Test function
    g.bench_function("levy 02", |b| b.iter(|| levy(black_box(P2))));
    g.bench_function("levy 10", |b| b.iter(|| levy(black_box(P10))));
    g.bench_function("levy 20", |b| b.iter(|| levy(black_box(P20))));
    // Derivative
    g.bench_function("levy_derivative 02", |b| {
        b.iter(|| levy_derivative(black_box(P2)))
    });
    g.bench_function("levy_derivative 10", |b| {
        b.iter(|| levy_derivative(black_box(P10)))
    });
    g.bench_function("levy_derivative 20", |b| {
        b.iter(|| levy_derivative(black_box(P20)))
    });
    // Derivative const
    g.bench_function("levy_derivative_const 02", |b| {
        b.iter(|| levy_derivative_const(black_box(P2)))
    });
    g.bench_function("levy_derivative_const 10", |b| {
        b.iter(|| levy_derivative_const(black_box(P10)))
    });
    g.bench_function("levy_derivative_const 20", |b| {
        b.iter(|| levy_derivative_const(black_box(P20)))
    });
    // Hessian
    g.bench_function("levy_hessian 02", |b| {
        b.iter(|| levy_hessian(black_box(P2)))
    });
    g.bench_function("levy_hessian 10", |b| {
        b.iter(|| levy_hessian(black_box(P10)))
    });
    g.bench_function("levy_hessian 20", |b| {
        b.iter(|| levy_hessian(black_box(P20)))
    });
    // Hessian const
    g.bench_function("levy_hessian_const 02", |b| {
        b.iter(|| levy_hessian_const(black_box(P2)))
    });
    g.bench_function("levy_hessian_const 10", |b| {
        b.iter(|| levy_hessian_const(black_box(P10)))
    });
    g.bench_function("levy_hessian_const 20", |b| {
        b.iter(|| levy_hessian_const(black_box(P20)))
    });
    g.finish();
}

pub fn bm_levy_n13(c: &mut Criterion) {
    let mut g = c.benchmark_group("levy_n13");
    // Test function
    g.bench_function("levy_n13 2", |b| b.iter(|| levy_n13(black_box(P2))));
    // Derivative
    g.bench_function("levy_n13_derivative 2", |b| {
        b.iter(|| levy_n13_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("levy_n13_hessian 2", |b| {
        b.iter(|| levy_n13_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_matyas(c: &mut Criterion) {
    let mut g = c.benchmark_group("matyas");
    // Test function
    g.bench_function("matyas 2", |b| b.iter(|| matyas(black_box(P2))));
    // Derivative
    g.bench_function("matyas_derivative 2", |b| {
        b.iter(|| matyas_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("matyas_hessian 2", |b| {
        b.iter(|| matyas_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_mccorminck(c: &mut Criterion) {
    let mut g = c.benchmark_group("mccorminck");
    // Test function
    g.bench_function("mccorminck 2", |b| b.iter(|| mccorminck(black_box(P2))));
    // Derivative
    g.bench_function("mccorminck_derivative 2", |b| {
        b.iter(|| mccorminck_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("mccorminck_hessian 2", |b| {
        b.iter(|| mccorminck_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_picheny(c: &mut Criterion) {
    let mut g = c.benchmark_group("picheny");
    // Test function
    g.bench_function("picheny 2", |b| b.iter(|| picheny(black_box(P2))));
    // Derivative
    g.bench_function("picheny_derivative 2", |b| {
        b.iter(|| picheny_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("picheny_hessian 2", |b| {
        b.iter(|| picheny_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_rastrigin(c: &mut Criterion) {
    let mut g = c.benchmark_group("rastrigin");
    // Test function
    g.bench_function("rastrigin 02", |b| b.iter(|| rastrigin(black_box(P2))));
    g.bench_function("rastrigin 10", |b| b.iter(|| rastrigin(black_box(P10))));
    g.bench_function("rastrigin 20", |b| b.iter(|| rastrigin(black_box(P20))));
    // Derivative
    g.bench_function("rastrigin_derivative 02", |b| {
        b.iter(|| rastrigin_derivative(black_box(P2)))
    });
    g.bench_function("rastrigin_derivative 10", |b| {
        b.iter(|| rastrigin_derivative(black_box(P10)))
    });
    g.bench_function("rastrigin_derivative 20", |b| {
        b.iter(|| rastrigin_derivative(black_box(P20)))
    });
    // Derivative const
    g.bench_function("rastrigin_derivative_const 02", |b| {
        b.iter(|| rastrigin_derivative_const(black_box(P2)))
    });
    g.bench_function("rastrigin_derivative_const 10", |b| {
        b.iter(|| rastrigin_derivative_const(black_box(P10)))
    });
    g.bench_function("rastrigin_derivative_const 20", |b| {
        b.iter(|| rastrigin_derivative_const(black_box(P20)))
    });
    // Hessian
    g.bench_function("rastrigin_hessian 02", |b| {
        b.iter(|| rastrigin_hessian(black_box(P2)))
    });
    g.bench_function("rastrigin_hessian 10", |b| {
        b.iter(|| rastrigin_hessian(black_box(P10)))
    });
    g.bench_function("rastrigin_hessian 20", |b| {
        b.iter(|| rastrigin_hessian(black_box(P20)))
    });
    // Hessian const
    g.bench_function("rastrigin_hessian_const 02", |b| {
        b.iter(|| rastrigin_hessian_const(black_box(P2)))
    });
    g.bench_function("rastrigin_hessian_const 10", |b| {
        b.iter(|| rastrigin_hessian_const(black_box(P10)))
    });
    g.bench_function("rastrigin_hessian_const 20", |b| {
        b.iter(|| rastrigin_hessian_const(black_box(P20)))
    });
    g.finish();
}

pub fn bm_rosenbrock(c: &mut Criterion) {
    let mut g = c.benchmark_group("rosenbrock");
    // Test function
    g.bench_function("rosenbrock 02", |b| b.iter(|| rosenbrock(black_box(P2))));
    g.bench_function("rosenbrock 10", |b| b.iter(|| rosenbrock(black_box(P10))));
    g.bench_function("rosenbrock 20", |b| b.iter(|| rosenbrock(black_box(P20))));
    // Derivative
    g.bench_function("rosenbrock_derivative 02", |b| {
        b.iter(|| rosenbrock_derivative(black_box(P2)))
    });
    g.bench_function("rosenbrock_derivative 10", |b| {
        b.iter(|| rosenbrock_derivative(black_box(P10)))
    });
    g.bench_function("rosenbrock_derivative 20", |b| {
        b.iter(|| rosenbrock_derivative(black_box(P20)))
    });
    // Derivative const
    g.bench_function("rosenbrock_derivative_const 02", |b| {
        b.iter(|| rosenbrock_derivative_const(black_box(P2)))
    });
    g.bench_function("rosenbrock_derivative_const 10", |b| {
        b.iter(|| rosenbrock_derivative_const(black_box(P10)))
    });
    g.bench_function("rosenbrock_derivative_const 20", |b| {
        b.iter(|| rosenbrock_derivative_const(black_box(P20)))
    });
    // Hessian
    g.bench_function("rosenbrock_hessian 02", |b| {
        b.iter(|| rosenbrock_hessian(black_box(P2)))
    });
    g.bench_function("rosenbrock_hessian 10", |b| {
        b.iter(|| rosenbrock_hessian(black_box(P10)))
    });
    g.bench_function("rosenbrock_hessian 20", |b| {
        b.iter(|| rosenbrock_hessian(black_box(P20)))
    });
    // Hessian const
    g.bench_function("rosenbrock_hessian_const 02", |b| {
        b.iter(|| rosenbrock_hessian_const(black_box(P2)))
    });
    g.bench_function("rosenbrock_hessian_const 10", |b| {
        b.iter(|| rosenbrock_hessian_const(black_box(P10)))
    });
    g.bench_function("rosenbrock_hessian_const 20", |b| {
        b.iter(|| rosenbrock_hessian_const(black_box(P20)))
    });
    g.finish();
}

pub fn bm_schaffer_n2(c: &mut Criterion) {
    let mut g = c.benchmark_group("schaffer_n2");
    // Test function
    g.bench_function("schaffer_n2 2", |b| b.iter(|| schaffer_n2(black_box(P2))));
    // Derivative
    g.bench_function("schaffer_n2_derivative 2", |b| {
        b.iter(|| schaffer_n2_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("schaffer_n2_hessian 2", |b| {
        b.iter(|| schaffer_n2_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_schaffer_n4(c: &mut Criterion) {
    let mut g = c.benchmark_group("schaffer_n4");
    // Test function
    g.bench_function("schaffer_n4 2", |b| b.iter(|| schaffer_n4(black_box(P2))));
    // Derivative
    g.bench_function("schaffer_n4_derivative 2", |b| {
        b.iter(|| schaffer_n4_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("schaffer_n4_hessian 2", |b| {
        b.iter(|| schaffer_n4_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_sphere(c: &mut Criterion) {
    let mut g = c.benchmark_group("sphere");
    // Test function
    g.bench_function("sphere 02", |b| b.iter(|| sphere(black_box(P2))));
    g.bench_function("sphere 10", |b| b.iter(|| sphere(black_box(P10))));
    g.bench_function("sphere 20", |b| b.iter(|| sphere(black_box(P20))));
    // Derivative
    g.bench_function("sphere_derivative 02", |b| {
        b.iter(|| sphere_derivative(black_box(P2)))
    });
    g.bench_function("sphere_derivative 10", |b| {
        b.iter(|| sphere_derivative(black_box(P10)))
    });
    g.bench_function("sphere_derivative 20", |b| {
        b.iter(|| sphere_derivative(black_box(P20)))
    });
    // Derivative const
    g.bench_function("sphere_derivative_const 02", |b| {
        b.iter(|| sphere_derivative_const(black_box(P2)))
    });
    g.bench_function("sphere_derivative_const 10", |b| {
        b.iter(|| sphere_derivative_const(black_box(P10)))
    });
    g.bench_function("sphere_derivative_const 20", |b| {
        b.iter(|| sphere_derivative_const(black_box(P20)))
    });
    // Hessian
    g.bench_function("sphere_hessian 02", |b| {
        b.iter(|| sphere_hessian(black_box(P2)))
    });
    g.bench_function("sphere_hessian 10", |b| {
        b.iter(|| sphere_hessian(black_box(P10)))
    });
    g.bench_function("sphere_hessian 20", |b| {
        b.iter(|| sphere_hessian(black_box(P20)))
    });
    // Hessian
    g.bench_function("sphere_hessian_const 02", |b| {
        b.iter(|| sphere_hessian_const(black_box(P2)))
    });
    g.bench_function("sphere_hessian_const 10", |b| {
        b.iter(|| sphere_hessian_const(black_box(P10)))
    });
    g.bench_function("sphere_hessian_const 20", |b| {
        b.iter(|| sphere_hessian_const(black_box(P20)))
    });
    g.finish();
}

pub fn bm_styblinski_tang(c: &mut Criterion) {
    let mut g = c.benchmark_group("styblinski_tang");
    // Test function
    g.bench_function("styblinski_tang 02", |b| {
        b.iter(|| styblinski_tang(black_box(P2)))
    });
    g.bench_function("styblinski_tang 10", |b| {
        b.iter(|| styblinski_tang(black_box(P10)))
    });
    g.bench_function("styblinski_tang 20", |b| {
        b.iter(|| styblinski_tang(black_box(P20)))
    });
    // Derivative
    g.bench_function("styblinski_tang_derivative 02", |b| {
        b.iter(|| styblinski_tang_derivative(black_box(P2)))
    });
    g.bench_function("styblinski_tang_derivative 10", |b| {
        b.iter(|| styblinski_tang_derivative(black_box(P10)))
    });
    g.bench_function("styblinski_tang_derivative 20", |b| {
        b.iter(|| styblinski_tang_derivative(black_box(P20)))
    });
    // Derivative const
    g.bench_function("styblinski_tang_derivative_const 02", |b| {
        b.iter(|| styblinski_tang_derivative_const(black_box(P2)))
    });
    g.bench_function("styblinski_tang_derivative_const 10", |b| {
        b.iter(|| styblinski_tang_derivative_const(black_box(P10)))
    });
    g.bench_function("styblinski_tang_derivative_const 20", |b| {
        b.iter(|| styblinski_tang_derivative_const(black_box(P20)))
    });
    // Hessian
    g.bench_function("styblinski_tang_hessian 02", |b| {
        b.iter(|| styblinski_tang_hessian(black_box(P2)))
    });
    g.bench_function("styblinski_tang_hessian 10", |b| {
        b.iter(|| styblinski_tang_hessian(black_box(P10)))
    });
    g.bench_function("styblinski_tang_hessian 20", |b| {
        b.iter(|| styblinski_tang_hessian(black_box(P20)))
    });
    // Hessian
    g.bench_function("styblinski_tang_hessian_const 02", |b| {
        b.iter(|| styblinski_tang_hessian_const(black_box(P2)))
    });
    g.bench_function("styblinski_tang_hessian_const 10", |b| {
        b.iter(|| styblinski_tang_hessian_const(black_box(P10)))
    });
    g.bench_function("styblinski_tang_hessian_const 20", |b| {
        b.iter(|| styblinski_tang_hessian_const(black_box(P20)))
    });
    g.finish();
}

pub fn bm_threehumpcamel(c: &mut Criterion) {
    let mut g = c.benchmark_group("threehumpcamel");
    // Test function
    g.bench_function("threehumpcamel 2", |b| {
        b.iter(|| threehumpcamel(black_box(P2)))
    });
    // Derivative
    g.bench_function("threehumpcamel_derivative 2", |b| {
        b.iter(|| threehumpcamel_derivative(black_box(P2)))
    });
    // Hessian
    g.bench_function("threehumpcamel_hessian 2", |b| {
        b.iter(|| threehumpcamel_hessian(black_box(P2)))
    });
    g.finish();
}

pub fn bm_zero(c: &mut Criterion) {
    let mut g = c.benchmark_group("zero");
    // Test function
    g.bench_function("zero 02", |b| b.iter(|| zero(black_box(P2))));
    g.bench_function("zero 10", |b| b.iter(|| zero(black_box(P10))));
    g.bench_function("zero 20", |b| b.iter(|| zero(black_box(P20))));
    // Derivative
    g.bench_function("zero_derivative 02", |b| {
        b.iter(|| zero_derivative(black_box(P2)))
    });
    g.bench_function("zero_derivative 10", |b| {
        b.iter(|| zero_derivative(black_box(P10)))
    });
    g.bench_function("zero_derivative 20", |b| {
        b.iter(|| zero_derivative(black_box(P20)))
    });
    // Derivative const
    g.bench_function("zero_derivative_const 02", |b| {
        b.iter(|| zero_derivative_const(black_box(P2)))
    });
    g.bench_function("zero_derivative_const 10", |b| {
        b.iter(|| zero_derivative_const(black_box(P10)))
    });
    g.bench_function("zero_derivative_const 20", |b| {
        b.iter(|| zero_derivative_const(black_box(P20)))
    });
    // Hessian
    g.bench_function("zero_hessian 02", |b| {
        b.iter(|| zero_hessian(black_box(P2)))
    });
    g.bench_function("zero_hessian 10", |b| {
        b.iter(|| zero_hessian(black_box(P10)))
    });
    g.bench_function("zero_hessian 20", |b| {
        b.iter(|| zero_hessian(black_box(P20)))
    });
    // Hessian
    g.bench_function("zero_hessian_const 02", |b| {
        b.iter(|| zero_hessian_const(black_box(P2)))
    });
    g.bench_function("zero_hessian_const 10", |b| {
        b.iter(|| zero_hessian_const(black_box(P10)))
    });
    g.bench_function("zero_hessian_const 20", |b| {
        b.iter(|| zero_hessian_const(black_box(P20)))
    });
    g.finish();
}

criterion_group!(
    benches,
    bm_ackley,
    bm_beale,
    bm_booth,
    bm_bukin_n6,
    bm_cross_in_tray,
    bm_easom,
    bm_eggholder,
    bm_goldsteinprice,
    bm_himmelblau,
    bm_holder_table,
    bm_levy,
    bm_levy_n13,
    bm_matyas,
    bm_mccorminck,
    bm_picheny,
    bm_rastrigin,
    bm_rosenbrock,
    bm_schaffer_n2,
    bm_schaffer_n4,
    bm_sphere,
    bm_styblinski_tang,
    bm_threehumpcamel,
    bm_zero,
);
criterion_main!(benches);
