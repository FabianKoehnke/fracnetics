#ifndef GYMNASIUM_WRAPPER_HPP
#define GYMNASIUM_WRAPPER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>

namespace py = pybind11;
using namespace py::literals;

class GymEnvWrapper {
private:
    py::object env;

public:
    GymEnvWrapper() = default;
    explicit GymEnvWrapper(const py::object& env_obj) : env(env_obj) {}

    py::tuple reset(int seed = -1, float curriculumLevel = 1.0f) {
        py::tuple reset_out;
        
        // 1. Standard Reset via Gymnasium
        if (seed >= 0) {
            reset_out = env.attr("reset")("seed"_a = seed);
        } else {
            reset_out = env.attr("reset")();
        }

        // 2. Wenn Curriculum aktiviert ist (< 1.0), Startzustand säubern & ausrichten
        if (curriculumLevel < 1.0f) {
            float level = std::max(0.0f, std::min(1.0f, curriculumLevel));
            py::object unwrapped = env.attr("unwrapped");
            
            if (py::hasattr(unwrapped, "lander")) {
                py::object lander = unwrapped.attr("lander");
                
                if (!lander.is_none()) {
                    // A) Box2D Force-Akkumulatoren leeren
                    if (py::hasattr(unwrapped, "world")) {
                        unwrapped.attr("world").attr("ClearForces")();
                    }

                    constexpr float FIXED_CENTER_X = 10.0f;
                    constexpr float TARGET_HEIGHT_ABOVE_PAD = 10.0f;

                    // Helipad-Höhe auslesen
                    float helipad_y = 3.333333f;
                    if (py::hasattr(unwrapped, "helipad_y")) {
                        helipad_y = unwrapped.attr("helipad_y").cast<float>();
                    }

                    // Aktuelle Zufallsposition auslesen
                    py::object pos = lander.attr("position");
                    float raw_x = pos.attr("x").cast<float>();
                    float raw_y = pos.attr("y").cast<float>();

                    // Zielposition & -winkel für den Rumpf berechnen
                    float target_x = FIXED_CENTER_X + (raw_x - FIXED_CENTER_X) * level;
                    float base_y = helipad_y + TARGET_HEIGHT_ABOVE_PAD;
                    float target_y = base_y + (raw_y - base_y) * level;
                    float target_angle = lander.attr("angle").cast<float>() * level;

                    // Atomare Transformation des Rumpfes via PyBox2D 'transform' Property
                    lander.attr("transform") = py::make_tuple(py::make_tuple(target_x, target_y), target_angle);

                    // Rumpf-Geschwindigkeiten skalieren
                    py::object linVel = lander.attr("linearVelocity");
                    float vx = linVel.attr("x").cast<float>() * level;
                    float vy = linVel.attr("y").cast<float>() * level;
                    lander.attr("linearVelocity") = py::make_tuple(vx, vy);
                    lander.attr("angularVelocity") = lander.attr("angularVelocity").cast<float>() * level;

                    // B) Beine (legs) trigonometrisch korrekt und ohne Gelenkspannung ausrichten
                    if (py::hasattr(unwrapped, "legs")) {
                        py::list legs = unwrapped.attr("legs");
                        constexpr float LEG_AWAY = 20.0f / 30.0f;  // ~0.667 Box2D Einheiten
                        constexpr float LEG_DOWN = -18.0f / 30.0f; // ~ -0.600 Box2D Einheiten
                        
                        float leg_x_offsets[2] = {-LEG_AWAY, LEG_AWAY};
                        float leg_angle_offsets[2] = {-0.05f, 0.05f}; // Relative Beinwinkel aus Gym

                        float cos_a = std::cos(target_angle);
                        float sin_a = std::sin(target_angle);

                        for (size_t i = 0; i < legs.size(); ++i) {
                            py::object leg = legs[i];
                            
                            // Exakte Rotation der Ankerpunkte mitdrehen
                            float off_x = leg_x_offsets[i];
                            float leg_target_x = target_x + (off_x * cos_a - LEG_DOWN * sin_a);
                            float leg_target_y = target_y + (off_x * sin_a + LEG_DOWN * cos_a);
                            
                            // Relativen Beinwinkel beibehalten
                            float leg_target_angle = target_angle + leg_angle_offsets[i];

                            // Atomare Transformation des Beins
                            leg.attr("transform") = py::make_tuple(py::make_tuple(leg_target_x, leg_target_y), leg_target_angle);

                            // Geschwindigkeiten der Beine anpassen
                            py::object legLinVel = leg.attr("linearVelocity");
                            float lvx = legLinVel.attr("x").cast<float>() * level;
                            float lvy = legLinVel.attr("y").cast<float>() * level;
                            leg.attr("linearVelocity") = py::make_tuple(lvx, lvy);
                            leg.attr("angularVelocity") = leg.attr("angularVelocity").cast<float>() * level;
                        }
                    }

                    // C) Neue Observation aus Gymnasium abfragen
                    py::object new_obs;
                    if (py::hasattr(unwrapped, "_get_obs")) {
                        new_obs = unwrapped.attr("_get_obs")();
                    } else if (py::hasattr(unwrapped, "_get_observation")) {
                        new_obs = unwrapped.attr("_get_observation")();
                    } else {
                        new_obs = reset_out[0];
                    }

                    return py::make_tuple(new_obs, reset_out[1]);
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
