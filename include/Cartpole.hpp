#include <cmath>
#include <random>
#include <iostream>

class CartPole {
public:
    double x = 0.0;
    double x_dot = 0.0;
    double theta = 0.0;
    double theta_dot = 0.0;
    std::vector<std::reference_wrapper<double>> observation = {x, x_dot, theta, theta_dot};

    const double gravity = 9.8;
    const double masscart = 1.0;
    const double masspole = 0.1;
    const double total_mass = masspole + masscart;
    const double length = 0.5;  // half-length
    const double polemass_length = masspole * length;
    const double force_mag = 10.0;
    const double tau = 0.02;  // seconds between state updates

    const double theta_threshold_radians = 12 * 2 * M_PI / 360;
    const double x_threshold = 2.4;

    bool done = false;

    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist{-0.01, 0.01};

    void reset() {
        x = dist(rng);
        x_dot = dist(rng);
        theta = dist(rng);
        theta_dot = dist(rng);
        done = false;
    }

    double step(int action) {
        // 0 = left, 1 = right
        double force = (action == 1) ? force_mag : -force_mag;
        double costheta = std::cos(theta);
        double sintheta = std::sin(theta);

        double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        double thetaacc = (gravity * sintheta - costheta * temp) /
                          (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        x += tau * x_dot;
        x_dot += tau * xacc;
        theta += tau * theta_dot;
        theta_dot += tau * thetaacc;

        // Check termination
        done = (x < -x_threshold || x > x_threshold ||
                theta < -theta_threshold_radians || theta > theta_threshold_radians);

        // Reward is always +1
        return 1.0;
    }

    void print_state() {
        std::cout << "x=" << x << ", x_dot=" << x_dot
                  << ", theta=" << theta << ", theta_dot=" << theta_dot << "\n";
    }
};

