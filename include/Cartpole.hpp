#include <cmath>
#include <random>
#include <iostream>
#include <array>

#ifndef M_PI
/** @cond INTERNAL */
#define M_PI 3.14159265358979323846
/** @endcond */
#endif

/**
 * @brief Simulates the classic CartPole environment (like Gymnasium).
 *  https://gymnasium.farama.org/environments/classic_control/cart_pole/
 *
 * The system state consists of four variables:
 *  - cart position
 *  - cart velocity
 *  - pole angle
 *  - pole angular velocity
 *
 * The class provides functionality to reset the environment,
 * take a step given an action, and access physical constants.
 */
class CartPole {
public:

    std::array<double, 4> state;

    // Physical constants (matching Gymnasium implementation)
    const double gravity = 9.8;
    const double masscart = 1.0;
    const double masspole = 0.1;
    const double total_mass = masspole + masscart;
    const double length = 0.5;  
    const double polemass_length = masspole * length;
    const double force_mag = 10.0;
    const double tau = 0.02;// seconds between state updates

    // Termination thresholds
    const double theta_threshold_radians = 12 * 2 * M_PI / 360; // 12 degrees
    const double x_threshold = 2.4;

    // Random number generator for state initialization
    std::shared_ptr<std::mt19937_64> rng;
    std::uniform_real_distribution<double> dist{-0.05, 0.05};

    /**
     * @brief Tracks whether the pole has just fallen or has been down
     * Steps beyond termination:
     *  -1 = pole not fallen
     *   0 = just fallen
     */
    int steps_beyond_terminated = -1;

    CartPole(std::shared_ptr<std::mt19937_64> _rng): 
        rng(_rng)
    {
        state = {0.0, 0.0, 0.0, 0.0};
    }

     /**
     * @brief Resets the environment to a random initial state.
     * @return Initial observation of the system
     */
    std::array<double, 4> reset() {
        state[0] = dist(*rng); // x
        state[1] = dist(*rng); // x_dot
        state[2] = dist(*rng); // theta
        state[3] = dist(*rng); // theta_dot

        steps_beyond_terminated = -1; 

        return state;
    }

    /*
     * @brief Struct encapsulating the result of a step() call.
     */

    // @cond INTERNAL 
    struct StepResult {
        std::array<double, 4> observation;
        double reward;
        bool terminated;
        bool truncated;
    };
    // @endcond

    /**
     * @brief Advances the environment by one time step.
     * @param action The action to take: 0 = push left, 1 = push right
     * @return StepResult containing new state, reward, termination, and truncation info
     */
    StepResult step(int action) {
        if (action != 0 && action != 1) {
            std::cerr << "Invalid action: " << action << " (must be 0 or 1)" << std::endl;
            action = 0; ///< Fallback to prevent crash
        }
        
        if (steps_beyond_terminated != -1) {
            return {state, 0.0, true, false}; 
        }

        double x = state[0];
        double x_dot = state[1];
        double theta = state[2];
        double theta_dot = state[3];

        double force = (action == 1) ? force_mag : -force_mag;
        double costheta = std::cos(theta);
        double sintheta = std::sin(theta);

        // Physics equations
        double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        double thetaacc = (gravity * sintheta - costheta * temp) /
                          (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        x = x + tau * x_dot;
        theta = theta + tau * theta_dot;
        x_dot = x_dot + tau * xacc;
        theta_dot = theta_dot + tau * thetaacc;

        state[0] = x;
        state[1] = x_dot;
        state[2] = theta;
        state[3] = theta_dot;

        // Check termination conditions
        bool terminated = (x < -x_threshold || x > x_threshold ||
                           theta < -theta_threshold_radians || theta > theta_threshold_radians);
        
        // Reward function (+1 per step until termination)
        double reward = 1.0;
        if (terminated) {
            if (steps_beyond_terminated == -1) {
                steps_beyond_terminated = 0;
            } else {
                steps_beyond_terminated++;
                reward = 0.0;
            }
        } else {
            steps_beyond_terminated = -1; ///< Reset if pole is upright
        }

        bool truncated_flag = false; 

        return {state, reward, terminated, truncated_flag};
    }

    /**
     * @brief Prints the current state to stdout.
     */
    void print_state() {
        std::cout << "x=" << state[0] << ", x_dot=" << state[1]
                  << ", theta=" << state[2] << ", theta_dot=" << state[3] << "\n";
    }
};

