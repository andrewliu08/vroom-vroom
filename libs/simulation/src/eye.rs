use std::f64::consts::PI;

use nalgebra as na;

use crate::food::Food;

pub struct Eye {
    pub(crate) fov_range: f64,
    pub(crate) fov_angle: f64,
    pub(crate) receptors: usize,
}

impl Eye {
    pub fn new(fov_range: f64, fov_angle: f64, receptors: usize) -> Self {
        Self {
            fov_range,
            fov_angle,
            receptors,
        }
    }

    pub fn default() -> Self {
        Self {
            fov_range: 0.5,
            fov_angle: PI / 2.0,
            receptors: 10,
        }
    }

    pub fn process_vision(
        &self,
        position: na::Point2<f64>,
        rotation: na::Rotation2<f64>,
        food: &[Food],
    ) -> Vec<f64> {
        let angle_per_receptor = self.fov_angle / self.receptors as f64;
        let mut receptors = vec![2.0; self.receptors];

        for f in food {
            let displacement = f.position - position;
            let dist = displacement.norm();
            if dist > self.fov_range {
                continue;
            }

            let angle = na::Rotation2::rotation_between(&na::Vector2::x(), &displacement).angle();
            let angle = na::wrap(angle - rotation.angle(), -PI, PI);
            let angle = angle + self.fov_angle / 2.0;
            if angle < 0.0 || angle > self.fov_angle {
                continue;
            }

            let receptor_idx =
                std::cmp::min((angle / angle_per_receptor) as usize, self.receptors - 1);
            receptors[receptor_idx] = f64::min(receptors[receptor_idx], dist / self.fov_range);
        }

        receptors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestCase {
        fov_range: f64,
        fov_angle: f64,
        receptors: usize,
        x: f64,
        y: f64,
        rotation: f64,
        food: Vec<Food>,
        expected: &'static str,
    }

    impl TestCase {
        fn run(&self) {
            let eye = Eye::new(self.fov_range, self.fov_angle, self.receptors);

            let actual = eye.process_vision(
                na::Point2::new(self.x, self.y),
                na::Rotation2::new(self.rotation),
                &self.food,
            );
            let actual = actual
                .into_iter()
                .map(|dist| {
                    if dist > 1.0 {
                        " "
                    } else if dist > 0.6 {
                        "."
                    } else if dist > 0.3 {
                        "o"
                    } else {
                        "O"
                    }
                })
                .collect::<Vec<_>>()
                .join("");

            assert_eq!(actual, self.expected);
        }
    }

    mod test_fov_ranges {
        use super::*;

        /*
             /......
           /........
        @>....o.....
           \........
             \......

             /....
           /......
        @>....o...
           \......
             \...

             /.
           /...
        @>....o
           \...
             \.

           /.
        @>... o
           \.
        */
        #[test]
        fn test() {
            let cases = [
                (1.0, "     O    "),
                (0.6, "     o    "),
                (0.2, "     .    "),
                (0.19, "          "),
            ];
            for (fov_range, expected) in cases {
                let food = vec![Food::new(na::Point2::new(0.2, 0.5))];
                TestCase {
                    fov_range,
                    fov_angle: PI / 2.0,
                    receptors: 10,
                    x: 0.0,
                    y: 0.5,
                    rotation: 0.0,
                    food,
                    expected,
                }
                .run();
            }
        }
    }

    mod test_fov_angles {
        use super::*;

        /*
            o
                o
        o   @>      o
                o
            o

            o
                o /...
        o   @>    ..o.
                o \...
            o

            o   /.....
               /o.....
        o   @>.....o..
               \o.....
            o   \.....

            o.........
            |...o.....
        o   @>.....o.
            |...o.....
            o.........

        ....o.........
        ........o.....
        o...@>.....o.
        ........o.....
        ....o.........
        */
        #[test]
        fn test() {
            let cases = [
                (0.0, "o         "),
                (PI / 180.0, "     o    "),
                (PI / 2.0, ".    o   ."),
                (PI, "o .  o . o"),
                (2.0 * PI, "  o. o.o o"),
            ];
            for (fov_angle, expected) in cases {
                let food = vec![
                    Food::new(na::Point2::new(1.0, 0.5)),
                    Food::new(na::Point2::new(1.0, 1.0)),
                    Food::new(na::Point2::new(1.0, 0.0)),
                    Food::new(na::Point2::new(0.5, 1.0)),
                    Food::new(na::Point2::new(0.5, 0.0)),
                    Food::new(na::Point2::new(0.0, 0.5)),
                ];
                TestCase {
                    fov_range: 1.0,
                    fov_angle,
                    receptors: 10,
                    x: 0.5,
                    y: 0.5,
                    rotation: 0.0,
                    food,
                    expected,
                }
                .run();
            }
        }
    }

    mod test_receptors {
        use super::*;

        /*
            |
            | o
            @>      o
            |
            |    o

            |
            | o
            @>------o
            |
            |    o

            |     /
            | o/
            @>      o
            |  \
            |    o\
        */
        #[test]
        fn test() {
            let cases = [(1, "O"), (2, "oO"), (3, "o.O")];
            for (receptors, expected) in cases {
                let food = vec![
                    Food::new(na::Point2::new(0.55, 0.6)),
                    Food::new(na::Point2::new(1.4, 0.5)),
                    Food::new(na::Point2::new(0.8, 0.1)),
                ];
                TestCase {
                    fov_range: 1.0,
                    fov_angle: PI,
                    receptors,
                    x: 0.5,
                    y: 0.5,
                    rotation: 0.0,
                    food,
                    expected,
                }
                .run();
            }
        }
    }

    mod test_position {
        use super::*;

        /*
              /..
            @>...
              \..
                o

              /..
            @>..o
              \..

            /....
          @>....o
            \....

                o
              /..
            @>...
              \..
        */
        #[test]
        fn test() {
            let cases = [
                (((0.5, 0.0), " ")),
                ((0.5, 0.5), "O"),
                ((0.2, 0.5), "."),
                ((0.5, 1.0), " "),
            ];
            for ((x, y), expected) in cases {
                let food = vec![Food::new(na::Point2::new(0.6, 0.5))];
                TestCase {
                    fov_range: 0.5,
                    fov_angle: PI / 2.0,
                    receptors: 1,
                    x,
                    y,
                    rotation: 0.0,
                    food,
                    expected,
                }
                .run();
            }
        }
    }

    mod test_rotation {
        use super::*;

        /*
            o
              /...
          o @>...o
              \...

          ..o..
          \ ^ /
          o @    o

            o
        ...\
        ..o<@    o
        .../
        */
        #[test]
        fn test() {
            let cases = [
                (0.0, "."),
                (PI / 2.0, "o"),
                (PI, "O"),
                (3.0 * PI / 2.0, " "),
            ];
            for (rotation, expected) in cases {
                let food = vec![
                    Food::new(na::Point2::new(1.4, 0.5)),
                    Food::new(na::Point2::new(0.5, 1.0)),
                    Food::new(na::Point2::new(0.4, 0.5)),
                ];
                TestCase {
                    fov_range: 1.0,
                    fov_angle: PI / 2.0,
                    receptors: 1,
                    x: 0.5,
                    y: 0.5,
                    rotation,
                    food,
                    expected,
                }
                .run();
            }
        }
    }
}
