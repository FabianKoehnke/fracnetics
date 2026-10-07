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

    py::tuple reset(int seed = -1, float curriculumLevel = 1.0f, bool absoluteImpulseCurriculum = false, bool validation = false,
                    bool uniformDirectionCurriculum = false, float directionAngle = 0.0f) {
        py::tuple reset_out;
        
        // 1. Standard Reset via Gymnasium
        if (seed >= 0) {
            reset_out = env.attr("reset")("seed"_a = seed);
        } else {
            reset_out = env.attr("reset")();
        }

        // 2. Curriculum: steuert den initialen Zufallsimpuls.
        //
        // Bei LunarLander-v3 ist die Spawn-Position und der Startwinkel des Landers
        // IMMER fix (siehe gymnasium/envs/box2d/lunar_lander.py: initial_x/initial_y
        // sind Konstanten, keine Zufallswerte). Die einzige Zufallsquelle, die die
        // Ausgangssituation eines Seeds schwer oder leicht macht, ist der einmalige
        // Kraftimpuls (ApplyForceToCenter mit uniform(-1000, 1000) je Achse). Da
        // Gymnasiums eigenes reset() intern bereits einen No-Op-Step ausfuehrt, ist
        // dieser Impuls zum Zeitpunkt von reset_out bereits in eine lineare/angulare
        // Geschwindigkeit umgesetzt.
        //
        // Zwei Modi stehen zur Verfuegung (absoluteImpulseCurriculum):
        //
        // - true ("absolute" Variante): Die Impuls-STAERKE wird direkt und
        //   absolut ueber curriculumLevel gesteuert (0.0 = kein Impuls, 1.0 = maximal
        //   kalibrierter Impuls MAX_PUSH_MAGNITUDE), UNABHAENGIG vom individuellen
        //   Seed. Nur die Richtung des Impulses bleibt zufaellig (aus dem natuerlichen
        //   Impuls dieses Seeds uebernommen). Dadurch trainieren am Anfang ALLE Seeds
        //   gleichermassen mit schwachem Impuls, keine Zufallsstreuung der Staerke
        //   mehr -> "am Anfang nur leichte Umgebungen, mit steigender Schwierigkeit
        //   staerkere Impulse".
        // - false (default; at curriculumLevel 1.0 Gymnasium's values pass through
        //   unchanged, "relative" Variante): Der seed-eigene Impuls (Staerke UND
        //   Richtung zufaellig) wird lediglich mit dem Faktor curriculumLevel
        //   herunterskaliert. Die relative Haerte-Reihenfolge zwischen Seeds bleibt
        //   erhalten, die absolute Staerke schwankt aber weiterhin mit dem Zufalls-Seed.
        //
        // WICHTIG: Die Erzwingung gilt fuer ALLE Trainings-Curriculum-Level von 0.0 bis
        // EINSCHLIESSLICH 1.0 (sonst wuerde ausgerechnet auf der letzten/haertesten
        // Trainingsstufe wieder die natuerliche, seed-abhaengig streuende Magnitude
        // durchschlagen -- ein Bruch mit dem Grundprinzip "gleiche Magnitude fuer alle
        // Seeds pro Level"). Nur fuer `validation == true` wird bewusst NICHT erzwungen,
        // damit Validierungslaeufe die echte, ungedeckelte Umgebung (inkl. der
        // natuerlichen Seed-zu-Seed-Streuung) widerspiegeln -- das ist der einzige Ort,
        // an dem wir absichtlich die reale Schwierigkeitsverteilung sehen wollen.
        if (!validation) {
            // Empirisch kalibriert: ueber den gesamten in diesem Projekt genutzten
            // Seed-Bereich (0-10000) liegt die natuerliche Impuls-Magnitude bei
            // max. ≈0.98 (p99.9 ≈0.955). MAX_PUSH_MAGNITUDE=1.05 liegt bewusst
            // etwas darueber, damit die letzte Trainingsstufe (Level 1.0) garantiert
            // mindestens so schwer ist wie der haerteste jemals gezogene reale Seed --
            // keine Luecke mehr zwischen Trainings- und Validierungsschwierigkeit.
            constexpr float MAX_PUSH_MAGNITUDE = 1.05f;
            // Empirisch kalibriert (0-10000 Seeds, obs[5] = 20*angularVelocity/FPS):
            // natuerliche |angularVelocity| liegt bei max. ≈0.183 (p99.9 ≈0.183).
            // Korrelation mit der linearen Impuls-Magnitude ist mit 0.88 hoch, aber
            // NICHT 1.0 -- angularVelocity ist also kein reiner Nebeneffekt der
            // linearen Skalierung und braucht einen EIGENEN, unabhaengigen Ziel-Betrag
            // pro Curriculum-Level (sonst bleibt die tatsaechliche resultierende
            // Winkelgeschwindigkeit nach der linearen Skalierung weiterhin seed-
            // abhaengig streuend). MAX_ANGULAR_PUSH_MAGNITUDE=0.20 liegt bewusst etwas
            // ueber dem natuerlichen Maximum, analog zu MAX_PUSH_MAGNITUDE.
            constexpr float MAX_ANGULAR_PUSH_MAGNITUDE = 0.20f;
            float level = std::max(0.0f, std::min(1.0f, curriculumLevel));
            py::object unwrapped = env.attr("unwrapped");

            if (py::hasattr(unwrapped, "lander")) {
                py::object lander = unwrapped.attr("lander");

                if (!lander.is_none()) {
                    py::object linVel = lander.attr("linearVelocity");
                    float vx = linVel.attr("x").cast<float>();
                    float vy = linVel.attr("y").cast<float>();

                    // WICHTIG (Unit-Fix): MAX_PUSH_MAGNITUDE ist in OBSERVATION-Einheiten
                    // kalibriert (siehe obs[2]/obs[3], analog analyzeSeedDifficulty()),
                    // nicht in rohen Box2D-Physik-Einheiten (letztere sind ca. 3-6x groesser
                    // und pro Achse asymmetrisch skaliert: obs_vx = vx*0.2, obs_vy =
                    // vy*0.1333). Wuerde man currentMag/targetMag stattdessen aus den rohen
                    // vx/vy berechnen, waere die resultierende Beobachtungs-Magnitude um
                    // genau diesen (Achsen-abhaengigen) Skalierungsfaktor verfaelscht -- das
                    // war der urspruengliche Bug. Wir berechnen currentMag daher aus der
                    // bereits korrekten Original-Observation (obs[2]/obs[3]), damit die
                    // erzwungene Ziel-Magnitude EXAKT im Observation-Raum stimmt, in dem
                    // das Netzwerk tatsaechlich rechnet.
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
                            scaleFactor = targetMag / currentMag; // Richtung bleibt erhalten, nur Betrag wird ersetzt
                        } else {
                            scaleFactor = 0.0f;
                        }

                        // Eigener, von der linearen Skalierung UNABHAENGIGER Ziel-Betrag
                        // fuer die Winkelgeschwindigkeit (siehe Kommentar oben zu
                        // MAX_ANGULAR_PUSH_MAGNITUDE). Vorzeichen (Rotationsrichtung)
                        // bleibt erhalten, nur der Betrag wird auf den Level-abhaengigen
                        // Zielwert gesetzt.
                        float currentAngMag = std::abs(obsAngVel);
                        float targetAngMag = MAX_ANGULAR_PUSH_MAGNITUDE * level;
                        if (currentAngMag > 1e-6f) {
                            angScaleFactor = targetAngMag / currentAngMag;
                        } else {
                            angScaleFactor = 0.0f;
                        }
                    } else {
                        scaleFactor = level; // alte relative Skalierung
                        angScaleFactor = level;
                    }
                    float newVx = vx * scaleFactor;
                    float newVy = vy * scaleFactor;

                    // Optional: erzwungene, GLEICHVERTEILTE Impuls-RICHTUNG statt der
                    // natuerlichen (quadratisch verteilten, siehe Kalibrierungsanalyse)
                    // Seed-Richtung. directionAngle wird pro Seed von aussen (Python)
                    // vorgegeben, z.B. stratifiziert ueber n Seeds eines Batches, damit
                    // jede Generation garantiert alle Richtungen gleichmaessig abdeckt --
                    // unabhaengig vom Zufalls-Seed. Nur mit absoluteImpulseCurriculum
                    // sinnvoll (sonst ist targetMag nicht definiert/seed-abhaengig).
                    //
                    // WICHTIG (Unit-Fix, analog zu obigem Magnitude-Fix): obs_vx/obs_vy
                    // sind ACHSEN-ASYMMETRISCH aus den rohen Box2D-Groessen skaliert
                    // (obs_vx = vx * (VIEWPORT_W/SCALE/2)/FPS, obs_vy = vy *
                    // (VIEWPORT_H/SCALE/2)/FPS -- die Konstanten unterscheiden sich, da
                    // Viewport-Breite und -Hoehe verschieden sind). directionAngle soll
                    // im OBSERVATION-Raum (dem Raum, in dem das Netzwerk tatsaechlich
                    // rechnet) gleichverteilt sein, NICHT im rohen Box2D-Raum. Wir leiten
                    // die beiden Achsen-Konstanten daher direkt aus dem natuerlichen
                    // (vx,vy)<->(obsVx,obsVy)-Paar dieses Resets ab (robuster als
                    // Gym-interne Konstanten hart zu kodieren) und setzen anschliessend
                    // NEUE Ziel-Rohgeschwindigkeiten, die im Observation-Raum exakt
                    // targetMag * (cos(angle), sin(angle)) ergeben.
                    float obsVxTarget = 0.0f;
                    float obsVyTarget = 0.0f;
                    bool directionForced = (absoluteImpulseCurriculum && uniformDirectionCurriculum);
                    if (directionForced) {
                        float targetMag = MAX_PUSH_MAGNITUDE * level;
                        constexpr float FALLBACK_CONST_X = 0.2f;    // (VIEWPORT_W/SCALE/2)/FPS, siehe lunar_lander.py
                        constexpr float FALLBACK_CONST_Y = 0.13333f; // (VIEWPORT_H/SCALE/2)/FPS, siehe lunar_lander.py
                        float constX = (std::abs(vx) > 1e-6f) ? (obsVx / vx) : FALLBACK_CONST_X;
                        float constY = (std::abs(vy) > 1e-6f) ? (obsVy / vy) : FALLBACK_CONST_Y;
                        obsVxTarget = std::cos(directionAngle) * targetMag;
                        obsVyTarget = std::sin(directionAngle) * targetMag;
                        newVx = obsVxTarget / constX;
                        newVy = obsVyTarget / constY;
                    }

                    lander.attr("linearVelocity") = py::make_tuple(newVx, newVy);
                    lander.attr("angularVelocity") = lander.attr("angularVelocity").cast<float>() * angScaleFactor;

                    // Beine haengen ueber ein Revolute-Joint am Rumpf; ihre TRANSLATORISCHE
                    // Geschwindigkeit wird konsistent mit scaleFactor (linear) mitskaliert,
                    // ihre ROTATORISCHE Geschwindigkeit mit angScaleFactor (Position/Winkel
                    // bleiben unveraendert, da der Lander seine Spawn-Position nach nur
                    // einem No-Op-Step kaum verlassen hat). Bei erzwungener Richtung wird
                    // fuer die Beine naeherungsweise weiterhin nur der Betrag (scaleFactor)
                    // angepasst -- die exakte Richtung der (kosmetisch untergeordneten)
                    // Beinbewegung ist fuer den Trainingsverlauf nicht entscheidend.
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

                    // WICHTIG (Bugfix): Diese Gymnasium-Version stellt KEINE aufrufbare
                    // _get_obs()/_get_observation()-Methode zur Verfuegung -- die
                    // Beobachtung wird stattdessen inline in step() aus vel.x/vel.y/
                    // angularVelocity berechnet (siehe lunar_lander.py). Ein fruehrer
                    // Versuch, diese (nicht existierenden) Methoden aufzurufen, ist
                    // STETS auf reset_out[0] (die UNVERAENDERTE Original-Beobachtung)
                    // zurueckgefallen -- die komplette Skalierung hatte dadurch bisher
                    // NIE einen sichtbaren Effekt auf die an das Netzwerk uebergebene
                    // erste Beobachtung, obwohl der Box2D-Koerper korrekt skaliert wurde!
                    //
                    // Fix: Da die Beobachtungskomponenten fuer vx/vy/angularVelocity
                    // LINEAR aus den jeweiligen Rohgroessen berechnet werden (state[2] =
                    // vel.x * const_x, state[3] = vel.y * const_y, state[5] = 20 *
                    // angularVelocity / FPS), gilt: wird die Rohgroesse mit scaleFactor
                    // multipliziert, skaliert sich die zugehoerige Beobachtungskomponente
                    // um EXAKT denselben Faktor -- unabhaengig von den genauen
                    // Normierungskonstanten. Wir muessen die Observation daher nicht neu
                    // generieren, sondern nur die 3 betroffenen Eintraege der bereits
                    // korrekten Original-Beobachtung nachtraeglich mit scaleFactor
                    // multiplizieren.
                    py::array_t<float> obsArray = reset_out[0].cast<py::array_t<float>>();
                    py::array_t<float> newObs(obsArray.size());
                    std::memcpy(newObs.mutable_data(), obsArray.data(), obsArray.size() * sizeof(float));
                    auto newObsPtr = newObs.mutable_data();
                    if (directionForced) {
                        // Bei erzwungener Richtung wurden obsVxTarget/obsVyTarget bereits
                        // EXAKT im Observation-Raum berechnet -- direkt uebernehmen statt
                        // ueber scaleFactor zu multiplizieren (waere hier nicht korrekt,
                        // da newVx/newVy nicht mehr proportional zu den natuerlichen vx/vy
                        // sind).
                        newObsPtr[2] = obsVxTarget;
                        newObsPtr[3] = obsVyTarget;
                    } else {
                        newObsPtr[2] *= scaleFactor; // vx-Komponente der Beobachtung
                        newObsPtr[3] *= scaleFactor; // vy-Komponente der Beobachtung
                    }
                    newObsPtr[5] *= angScaleFactor; // angularVelocity-Komponente der Beobachtung

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
