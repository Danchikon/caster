use once_cell::sync::Lazy;
use pyo3::prelude::PyModule;
use pyo3::{pyclass, pyfunction, pymethods, pymodule, wrap_pyfunction, PyResult, Python};
use threadpool::ThreadPool;

const NUM_THREADS: usize = 4;

static THREAD_POOL: Lazy<ThreadPool> = Lazy::new(|| ThreadPool::new(NUM_THREADS));

#[pyclass]
#[derive(Clone)]
struct Intersection {
    x: f32,
    y: f32,
    len: f32,
}

#[pymethods]
impl Intersection {
    #[new]
    fn new(x: f32, y: f32, len: f32) -> Self {
        Intersection { x, y, len }
    }

    #[getter]
    fn x(&self) -> f32 {
        self.x
    }

    #[getter]
    fn y(&self) -> f32 {
        self.y
    }

    #[getter]
    fn len(&self) -> f32 {
        self.len
    }
}

#[pyclass]
struct Ray {
    x: f32,
    y: f32,
    angle: f32,
    intersection: Option<Intersection>,
}

#[pymethods]
impl Ray {
    #[new]
    fn new(x: f32, y: f32, angle: f32, intersection: Option<Intersection>) -> Self {
        Self {
            x,
            y,
            angle,
            intersection,
        }
    }

    #[getter]
    fn x(&self) -> f32 {
        self.x
    }

    #[getter]
    fn y(&self) -> f32 {
        self.y
    }

    #[getter]
    fn angle(&self) -> f32 {
        self.angle
    }

    #[getter]
    fn intersection(&self) -> Option<Intersection> {
        match &self.intersection {
            Some(value) => Some(value.clone()),
            None => None,
        }
    }
}

#[pyfunction]
fn init() {
    Lazy::force(&THREAD_POOL);
}

#[pyfunction]
fn rays(
    view_angle: f32,
    fov: f32,
    rays_count: usize,
    ray_begin_x: f32,
    ray_begin_y: f32,
    ray_len: f32,
    lines: Vec<(f32, f32, f32, f32)>,
    circles: Vec<(f32, f32, f32)>,
    circles_accuracy: f32,
) -> PyResult<Vec<Ray>> {
    let mut rays = Vec::<Ray>::with_capacity(rays_count);

    let angle_offset = view_angle - fov / 2.0;

    for i in 0..rays_count {
        let angle = i as f32 / rays_count as f32 * fov + angle_offset;

        let intersection_result = intersection(
            ray_begin_x,
            ray_begin_y,
            ray_len,
            angle,
            lines.clone(),
            circles.clone(),
            circles_accuracy,
        );

        let intersection = match intersection_result {
            Ok(value) => value,
            Err(error) => return Err(error),
        };

        rays.push(Ray::new(ray_begin_x, ray_begin_y, angle, intersection));
    }

    Ok(rays)
}

#[pyfunction]
fn intersection(
    ray_begin_x: f32,
    ray_begin_y: f32,
    ray_len: f32,
    ray_angle: f32,
    lines: Vec<(f32, f32, f32, f32)>,
    circles: Vec<(f32, f32, f32)>,
    circles_accuracy: f32,
) -> PyResult<Option<Intersection>> {
    let intersect_lines_result =
        intersection_lines(ray_begin_x, ray_begin_y, ray_len, ray_angle, lines);

    let intersect_lines = match intersect_lines_result {
        Ok(value) => value,
        _ => return intersect_lines_result,
    };

    let intersect_circles_result = intersection_circles(
        ray_begin_x,
        ray_begin_y,
        ray_len,
        ray_angle,
        circles,
        circles_accuracy,
    );

    let intersect_circles = match intersect_circles_result {
        Ok(value) => value,
        _ => return intersect_circles_result,
    };

    let closest = choose_closest(intersect_lines, intersect_circles);

    return Ok(closest);
}

#[pyfunction]
fn intersection_lines(
    ray_begin_x: f32,
    ray_begin_y: f32,
    ray_len: f32,
    ray_angle: f32,
    lines: Vec<(f32, f32, f32, f32)>,
) -> PyResult<Option<Intersection>> {
    let mut closest: Option<Intersection> = None;

    for (line_x1, line_y1, line_x2, line_y2) in lines {
        let intersect_result = intersection_line(
            ray_begin_x,
            ray_begin_y,
            ray_len,
            ray_angle,
            line_x1,
            line_y1,
            line_x2,
            line_y2,
        );

        let intersect = match intersect_result {
            Ok(value) => value,
            _ => return intersect_result,
        };

        closest = choose_closest(closest, intersect)
    }

    return Ok(closest);
}

#[pyfunction]
fn intersection_line(
    ray_begin_x: f32,
    ray_begin_y: f32,
    ray_len: f32,
    ray_angle: f32,
    line_x1: f32,
    line_y1: f32,
    line_x2: f32,
    line_y2: f32,
) -> PyResult<Option<Intersection>> {
    let ray_end_x = ray_angle.cos() * ray_len + ray_begin_x;
    let ray_end_y = ray_angle.sin() * ray_len + ray_begin_y;

    let ray_x_diff = ray_end_x - ray_begin_x;
    let ray_tg = (ray_end_y - ray_begin_y) / ray_x_diff;
    let ray_b = ray_begin_y - ray_tg * ray_begin_x;

    let line_x_diff = line_x2 - line_x1;
    let line_tg = (line_y2 - line_y1) / line_x_diff;
    let tg_diff = line_tg - ray_tg;

    if tg_diff == 0f32 {
        return Ok(None);
    }

    let line_b = line_y1 - line_tg * line_x1;

    let y = (ray_b * line_tg - line_b * ray_tg) / tg_diff;
    let x = (y - ray_b) / ray_tg;

    if line_x1.min(line_x2) > x
        || x > line_x1.max(line_x2)
        || line_y1.min(line_y2) > y
        || y > line_y1.max(line_y2)
    {
        return Ok(None);
    }

    let len = vector_len(ray_begin_x, ray_begin_y, x, y);

    if len > ray_len {
        return Ok(None);
    }

    let cos = vectors_cos(
        ray_begin_x,
        ray_begin_y,
        ray_end_x,
        ray_end_y,
        x,
        y,
        len,
        ray_len,
    );

    if cos.is_sign_negative() {
        return Ok(None);
    }

    return Ok(Some(Intersection { x, y, len }));
}

#[pyfunction]
fn intersection_circles(
    ray_begin_x: f32,
    ray_begin_y: f32,
    ray_len: f32,
    ray_angle: f32,
    circles: Vec<(f32, f32, f32)>,
    accuracy: f32,
) -> PyResult<Option<Intersection>> {
    let mut closest: Option<Intersection> = None;

    for (circle_x, circle_y, circle_radius) in circles {
        let intersect_result = intersection_circle(
            ray_begin_x,
            ray_begin_y,
            ray_len,
            ray_angle,
            circle_x,
            circle_y,
            circle_radius,
            accuracy,
            0f32,
        );

        let intersect = match intersect_result {
            Ok(value) => value,
            _ => return intersect_result,
        };

        closest = choose_closest(closest, intersect);
    }

    return Ok(closest);
}

#[pyfunction]
fn intersection_circle(
    ray_begin_x: f32,
    ray_begin_y: f32,
    ray_len: f32,
    ray_angle: f32,
    circle_x: f32,
    circle_y: f32,
    circle_radius: f32,
    accuracy: f32,
    mut len: f32,
) -> PyResult<Option<Intersection>> {
    if len > ray_len {
        return Ok(None);
    }

    let len_to_center = vector_len(ray_begin_x, ray_begin_y, circle_x, circle_y);

    let len_to_circle = if len_to_center < circle_radius {
        circle_radius - len_to_center
    } else {
        len_to_center - circle_radius
    };

    if len_to_circle <= accuracy {
        return Ok(Some(Intersection {
            x: ray_begin_x,
            y: ray_begin_y,
            len,
        }));
    }

    let next_x = ray_angle.cos() * len_to_circle + ray_begin_x;
    let next_y = ray_angle.sin() * len_to_circle + ray_begin_y;
    len += len_to_circle;

    intersection_circle(
        next_x,
        next_y,
        ray_len,
        ray_angle,
        circle_x,
        circle_y,
        circle_radius,
        accuracy,
        len,
    )
}

fn vectors_cos(x: f32, y: f32, x1: f32, y1: f32, x2: f32, y2: f32, len1: f32, len2: f32) -> f32 {
    ((x - x1) * (x - x2) + (y - y1) * (y - y2)) / (len1 * len2)
}

fn vector_len(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
}

fn choose_closest(
    first: Option<Intersection>,
    second: Option<Intersection>,
) -> Option<Intersection> {
    match &first {
        Some(first_intersection) => match &second {
            Some(second_intersection) => {
                if first_intersection.len > second_intersection.len {
                    second
                } else {
                    first
                }
            }
            None => first,
        },
        None => second,
    }
}

#[pymodule]
fn caster(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<Intersection>().unwrap();
    module
        .add_function(wrap_pyfunction!(intersection, module).unwrap())
        .unwrap();
    module
        .add_function(wrap_pyfunction!(intersection_lines, module).unwrap())
        .unwrap();
    module
        .add_function(wrap_pyfunction!(intersection_line, module).unwrap())
        .unwrap();
    module
        .add_function(wrap_pyfunction!(intersection_circles, module).unwrap())
        .unwrap();
    module
        .add_function(wrap_pyfunction!(intersection_circle, module).unwrap())
        .unwrap();
    module
        .add_function(wrap_pyfunction!(rays, module).unwrap())
        .unwrap();
    module
        .add_function(wrap_pyfunction!(init, module).unwrap())
        .unwrap();
    Ok(())
}
