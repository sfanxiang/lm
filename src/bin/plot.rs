use plotpy::{Curve, Plot, StrError};
fn main() -> Result<(), StrError> {
    // generate the x-values (tokens) and y-values (rust and py times)
    let x = &[
        5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 90.,
        95., 100.,
    ];
    // Xiang's generated data
    /*let mut python_data = &[
        0.37010812759399414,
        0.6232638359069824,
        0.7645416259765625,
        0.8862431049346924,
        1.0581285953521729,
        1.2347609996795654,
        1.3949849605560303,
        1.5694243907928467,
        1.77116060256958,
        1.921870470046997,
        2.08957576751709,
        2.2649388313293457,
        2.424726963043213,
        2.5992777347564697,
        2.79337477684021,
        2.9743425846099854,
        3.144225835800171,
        3.356588125228882,
        3.504913330078125,
    ];
    */

    // Chris's archlinux generated data
    let python_data = &[
        0.30283427238464355,
        0.5330073833465576,
        0.7986268997192383,
        1.0692846775054932,
        1.286877155303955,
        1.5224871635437012,
        1.8031561374664307,
        2.0303425788879395,
        2.3067824840545654,
        2.618229389190674,
        2.8433303833007812,
        3.1177589893341064,
        3.371561288833618,
        3.640733003616333,
        3.8751988410949707,
        4.128981590270996,
        4.425241470336914,
        4.695688247680664,
        5.260429859161377,
        5.347133636474609,
    ];

    /*let rust_data = &[
        0.287, 0.5, 0.678, 0.719, 0.867, 1.012, 1.153, 1.304, 1.443, 1.597, 1.738, 1.888, 2.037,
         2.171, 2.328, 2.463, 2.611, 2.753, 2.911,
    ];*/
    let rust_data = &[
        0.335, 0.492, 0.737, 0.973, 1.181, 1.391, 1.639, 1.857, 2.096, 2.336, 2.567, 2.801, 3.033,
        3.259, 3.498, 3.74, 3.953, 4.183, 4.448, 4.839,
    ];

    // configure curves
    let mut curve_python = Curve::new();
    curve_python
        .set_label("Python")
        .set_line_alpha(0.8)
        .set_line_color("#4A6DBD")
        .set_line_style("-");

    let mut curve_rust = Curve::new();
    curve_rust
        .set_label("Rust")
        .set_line_alpha(0.8)
        .set_line_color("#FF6318")
        .set_line_style("-");

    // draw the python curve
    curve_python.draw(x, python_data);

    // draw the rust curve
    curve_rust.draw(x, rust_data);

    // add contour to plot
    let mut plot = Plot::new();
    plot.add(&curve_python);
    plot.add(&curve_rust);
    plot.set_labels("Number of New Tokens Generated", "Time [seconds]");
    plot.set_title("Speed of Generating Tokens in Rust vs Python");
    plot.legend();
    // save figure
    plot.save("Rust_v_Python.pdf")?;
    Ok(())
}
