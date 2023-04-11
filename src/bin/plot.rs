use plotpy::{Curve, Plot, StrError};
fn main() -> Result<(), StrError> {
    // generate the x-values (tokens) and y-values (rust and py times)
    let x = &[10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.];
    let python_data = &[
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
    let rust_data = &[
        0.287, 0.5, 0.678, 0.719, 0.867, 1.012, 1.153, 1.304, 1.443, 1.597, 1.738, 1.888, 2.037,
        2.171, 2.328, 2.463, 2.611, 2.753, 2.911,
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
    plot.save("../../tmp/Rust_v_Python.pdf")?;
    Ok(())
}
