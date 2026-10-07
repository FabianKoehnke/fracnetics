#ifndef GYMNASIUM_WRAPPER_HPP
#define GYMNASIUM_WRAPPER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace py = pybind11;
using namespace py::literals;

class GymEnvWrapper {
private:
    py::object env;

public:
    GymEnvWrapper() = default;
    explicit GymEnvWrapper(const py::object& env_obj) : env(env_obj) {}

    /**
     * @brief Resets the environment and optionally reshapes the initial random push (curriculum).
     *
     * @details
     * In LunarLander-v3 the spawn position and start angle are fixed (initial_x/initial_y are
     * constants in gymnasium/envs/box2d/lunar_lander.py). The only random source that makes a seed
     * easy or hard is the one-off push (ApplyForceToCenter with uniform(-1000, 1000) per axis). Since
     * Gymnasium's reset() already runs one no-op step, the push has turned into linear and angular
     * velocity by the time reset_out is returned; this method rescales those velocities.
     *
     * Two modes (absoluteImpulseCurriculum):
     * - false (default, "relative"): the seed's own push (strength AND direction random) is scaled by
     *   curriculumLevel. The difficulty order between seeds is kept, the absolute strength still
     *   varies with the seed. At curriculumLevel 1.0 Gymnasium's values pass through unchanged.
     * - true ("absolute"): the push STRENGTH is set by curriculumLevel alone (0.0 = no push,
     *   1.0 = MAX_PUSH_MAGNITUDE), independent of the seed; only the direction is the seed's own.
     *   All seeds start equally easy at low levels and get stronger pushes as difficulty rises. This
     *   also holds at level 1.0 -- otherwise the last stage would fall back to the natural,
     *   seed-dependent magnitude and break "same magnitude for all seeds per level".
     *
     * Calibration over seeds 0-10000: the natural push magnitude in observation units peaks at
     * ≈0.98 (p99.9 ≈0.955), so MAX_PUSH_MAGNITUDE = 1.05 makes level 1.0 at least as hard as the
     * hardest real seed. The natural |obs[5]| (= 20*angularVelocity/FPS) peaks at ≈0.183; its
     * correlation with the linear magnitude is 0.88, not 1.0, so the angular velocity gets its own
     * target MAX_ANGULAR_PUSH_MAGNITUDE = 0.20 per level instead of riding on the linear scaling.
     *
     * Magnitudes are computed in OBSERVATION units (obs[2]/obs[3]), the space the network works in.
     * Raw Box2D velocities are 3-6x larger and scaled asymmetrically per axis (obs_vx = vx * 0.2,
     * obs_vy = vy * 0.1333), so a magnitude computed from them would be distorted by that factor.
     *
     * With uniformDirectionCurriculum (absolute mode only) the push DIRECTION is forced to
     * directionAngle, uniform in observation space instead of the natural, non-uniform seed
     * direction. The caller (Python) sets the angle per seed, e.g. stratified over a batch. The two
     * axis constants are derived from this reset's (vx, vy) <-> (obs_vx, obs_vy) pair instead of
     * hard-coding Gymnasium internals.
     *
     * The legs hang on revolute joints: their linear velocity is scaled with the linear factor, their
     * angular velocity with the angular factor (positions stay, the lander has barely moved after one
     * step). With a forced direction the legs only get the magnitude factor -- an approximation that
     * does not matter for training.
     *
     * This Gymnasium version has no callable _get_obs(); the observation is built inline in step().
     * Since obs[2], obs[3] and obs[5] are linear in vel.x, vel.y and angularVelocity, the returned
     * observation is patched by the same factors instead of regenerated (with a forced direction the
     * targets are written directly). An earlier version fell back to the unchanged observation, so
     * the scaling never reached the network although the Box2D body was scaled.
     *
     * @param seed Seed for env.reset(); -1 resets without a seed.
     * @param curriculumLevel Push level in [0, 1] (clamped).
     * @param absoluteImpulseCurriculum Selects the mode, see above. Default false.
     * @param validation If true, nothing is changed: validation always sees the real environment,
     *        including the natural seed-to-seed spread.
     * @param uniformDirectionCurriculum Forces the push direction to directionAngle (absolute mode only).
     * @param directionAngle Push direction in radians, in observation space.
     * @return (observation, info) as returned by env.reset(), the observation patched if the start was changed.
     *
     * @note When the start is changed, Gymnasium has already computed its shaping baseline
     *       (prev_shaping) from the unchanged velocities, so the first step's reward does not match
     *       the changed start exactly.
     */
    py::tuple reset(int seed = -1, float curriculumLevel = 1.0f, bool absoluteImpulseCurriculum = false, bool validation = false,
                    bool uniformDirectionCurriculum = false, float directionAngle = 0.0f) {
        py::tuple reset_out;
        
        // 1. Standard Reset via Gymnasium
        if (seed >= 0) {
            reset_out = env.attr("reset")("seed"_a = seed);
        } else {
            reset_out = env.attr("reset")();
        }

        // 2. Curriculum: rescale the initial push (see docstring); validation stays untouched
        if (!validation) {
            // Calibrated on seeds 0-10000, in observation units (see docstring)
            constexpr float MAX_PUSH_MAGNITUDE = 1.05f;
            constexpr float MAX_ANGULAR_PUSH_MAGNITUDE = 0.20f;
            float level = std::max(0.0f, std::min(1.0f, curriculumLevel));
            py::object unwrapped = env.attr("unwrapped");

            if (py::hasattr(unwrapped, "lander")) {
                py::object lander = unwrapped.attr("lander");

                if (!lander.is_none()) {
                    py::object linVel = lander.attr("linearVelocity");
                    float vx = linVel.attr("x").cast<float>();
                    float vy = linVel.attr("y").cast<float>();

                    // Magnitudes in observation units, not raw Box2D velocities (see docstring)
                    py::array_t<float> obsArrayNatural = reset_out[0].cast<py::array_t<float>>();
                    float obsVx = obsArrayNatural.data()[2];
                    float obsVy = obsArrayNatural.data()[3];
                    float obsAngVel = obsArrayNatural.data()[5];

                    float scaleFactor;
                    float angScaleFactor;
                    if (absoluteImpulseCurriculum) {
                        float currentMag = std::sqrt(obsVx * obsVx + obsVy * obsVy);
                        float targetMag = MAX_PUSH_MAGNITUDE * level;
                        if (currentMag > 1e-6f) {
                            scaleFactor = targetMag / currentMag; // keep direction, replace magnitude
                        } else {
                            scaleFactor = 0.0f;
                        }

                        // Own target for the angular velocity; the sign is kept
                        float currentAngMag = std::abs(obsAngVel);
                        float targetAngMag = MAX_ANGULAR_PUSH_MAGNITUDE * level;
                        if (currentAngMag > 1e-6f) {
                            angScaleFactor = targetAngMag / currentAngMag;
                        } else {
                            angScaleFactor = 0.0f;
                        }
                    } else {
                        scaleFactor = level; // relative scaling
                        angScaleFactor = level;
                    }
                    float newVx = vx * scaleFactor;
                    float newVy = vy * scaleFactor;

                    // Optional forced direction, uniform in observation space (see docstring)
                    float obsVxTarget = 0.0f;
                    float obsVyTarget = 0.0f;
                    bool directionForced = (absoluteImpulseCurriculum && uniformDirectionCurriculum);
                    if (directionForced) {
                        float targetMag = MAX_PUSH_MAGNITUDE * level;
                        constexpr float FALLBACK_CONST_X = 0.2f;    // (VIEWPORT_W/SCALE/2)/FPS, see lunar_lander.py
                        constexpr float FALLBACK_CONST_Y = 0.13333f; // (VIEWPORT_H/SCALE/2)/FPS, see lunar_lander.py
                        float constX = (std::abs(vx) > 1e-6f) ? (obsVx / vx) : FALLBACK_CONST_X;
                        float constY = (std::abs(vy) > 1e-6f) ? (obsVy / vy) : FALLBACK_CONST_Y;
                        obsVxTarget = std::cos(directionAngle) * targetMag;
                        obsVyTarget = std::sin(directionAngle) * targetMag;
                        newVx = obsVxTarget / constX;
                        newVy = obsVyTarget / constY;
                    }

                    lander.attr("linearVelocity") = py::make_tuple(newVx, newVy);
                    lander.attr("angularVelocity") = lander.attr("angularVelocity").cast<float>() * angScaleFactor;

                    // Legs: linear part with scaleFactor, angular part with angScaleFactor
                    if (py::hasattr(unwrapped, "legs")) {
                        py::list legs = unwrapped.attr("legs");
                        for (size_t i = 0; i < legs.size(); ++i) {
                            py::object leg = legs[i];
                            py::object legLinVel = leg.attr("linearVelocity");
                            float lvx = legLinVel.attr("x").cast<float>() * scaleFactor;
                            float lvy = legLinVel.attr("y").cast<float>() * scaleFactor;
                            leg.attr("linearVelocity") = py::make_tuple(lvx, lvy);
                            leg.attr("angularVelocity") = leg.attr("angularVelocity").cast<float>() * angScaleFactor;
                        }
                    }

                    // No callable _get_obs(): patch the linear obs entries by the same factors
                    py::array_t<float> obsArray = reset_out[0].cast<py::array_t<float>>();
                    py::array_t<float> newObs(obsArray.size());
                    std::memcpy(newObs.mutable_data(), obsArray.data(), obsArray.size() * sizeof(float));
                    auto newObsPtr = newObs.mutable_data();
                    if (directionForced) {
                        // Targets are already in observation space
                        newObsPtr[2] = obsVxTarget;
                        newObsPtr[3] = obsVyTarget;
                    } else {
                        newObsPtr[2] *= scaleFactor; // vx
                        newObsPtr[3] *= scaleFactor; // vy
                    }
                    newObsPtr[5] *= angScaleFactor; // angular velocity

                    return py::make_tuple(newObs, reset_out[1]);
                }
            }
        }

        return reset_out;
    }

    py::tuple step(const py::object& action) {
        return env.attr("step")(action);
    }

    py::tuple step(int action) {
        return step(py::int_(action));
    }

    void render() {
        env.attr("render")();
    }

    void close() {
        env.attr("close")();
    }
};

#endif // GYMNASIUM_WRAPPER_HPP
