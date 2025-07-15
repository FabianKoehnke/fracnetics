#include <cmath>
#include <random>
#include <iostream>
#include <array>

// Definition von M_PI, falls nicht automatisch von <cmath> bereitgestellt
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class CartPole {
public:
    // Zustand des CartPole-Systems: [cart position, cart velocity, pole angle, pole angular velocity]
    std::array<double, 4> state;

    // Konstanten wie im Gymnasium-Code
    const double gravity = 9.8;
    const double masscart = 1.0;
    const double masspole = 0.1;
    const double total_mass = masspole + masscart;
    const double length = 0.5;  // actually half the pole's length
    const double polemass_length = masspole * length;
    const double force_mag = 10.0;
    const double tau = 0.02;    // seconds between state updates

    // Grenzwerte für die Terminierung
    const double theta_threshold_radians = 12 * 2 * M_PI / 360; // 12 degrees
    const double x_threshold = 2.4;

    // Für die Zufallsinitialisierung
    std::shared_ptr<std::mt19937_64> rng; // Zufälliger Seed bei jedem Lauf, oder mit Seed gesetzt
    std::uniform_real_distribution<double> dist{-0.05, 0.05};

    // Dieser Wert wird in Gymnasium für das Belohnungssystem benötigt,
    // um zu wissen, ob der Pol gerade erst gefallen ist oder schon länger liegt.
    int steps_beyond_terminated = -1; // -1: Pol nicht gefallen, 0: gerade gefallen

    // Standard-Konstruktor
    CartPole(std::shared_ptr<std::mt19937_64> _rng): 
        rng(_rng)
    {
        state = {0.0, 0.0, 0.0, 0.0}; // Initialisiert den Zustand auf Nullen
    }

    // Konstruktor mit Seed für reproduzierbare Ergebnisse
    explicit CartPole() {
        state = {0.0, 0.0, 0.0, 0.0};
    }

    // Die reset-Funktion gibt die initiale Beobachtung zurück
    std::array<double, 4> reset() {
        // Generiert 4 Zufallswerte im Bereich [-0.05, 0.05]
        state[0] = dist(*rng); // x
        state[1] = dist(*rng); // x_dot
        state[2] = dist(*rng); // theta
        state[3] = dist(*rng); // theta_dot

        // Setzt die Terminierungsflags zurück
        steps_beyond_terminated = -1; 

        return state;
    }

    // Struktur zur Kapselung der Rückgabewerte von step()
    struct StepResult {
        std::array<double, 4> observation;
        double reward;
        bool terminated;
        bool truncated; // Für Gymnasium-Konformität
    };

    // Die step-Funktion führt einen Schritt in der Umgebung aus
    StepResult step(int action) {
        // Überprüfungen wie in Gymnasium
        if (action != 0 && action != 1) {
            std::cerr << "Ungültige Aktion: " << action << " (Aktion muss 0 oder 1 sein)" << std::endl;
            action = 0; // Fallback-Wert, um Absturz zu vermeiden
        }
        
        // Wenn die Umgebung bereits terminiert ist, geben wir nur den aktuellen Zustand zurück
        // und eine Belohnung von 0.0 (wie im Gymnasium nach dem ersten Terminierungsschritt)
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

        // Die Physik-Gleichungen
        double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        double thetaacc = (gravity * sintheta - costheta * temp) /
                          (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        // --- HIER IST DIE ENTSCHEIDENDE INTEGRATIONSREIHENFOLGE (Euler-Methode wie in Gymnasium) ---
        // Zuerst Positionen aktualisieren (mit alten Geschwindigkeiten)
        x = x + tau * x_dot;
        theta = theta + tau * theta_dot;
        // Dann Geschwindigkeiten aktualisieren
        x_dot = x_dot + tau * xacc;
        theta_dot = theta_dot + tau * thetaacc;

        // Speichern des neuen Zustands
        state[0] = x;
        state[1] = x_dot;
        state[2] = theta;
        state[3] = theta_dot;

        // Überprüfung der Terminierungsbedingungen
        bool terminated = (x < -x_threshold || x > x_threshold ||
                           theta < -theta_threshold_radians || theta > theta_threshold_radians);
        
        // Belohnungssystem (standardmäßig +1 für jeden Schritt bis zur Terminierung)
        double reward = 1.0;
        if (terminated) {
            if (steps_beyond_terminated == -1) {
                steps_beyond_terminated = 0;
            } else {
                steps_beyond_terminated++;
                reward = 0.0; // Nach dem ersten Terminierungsschritt ist die Belohnung 0.0
            }
        } else {
            steps_beyond_terminated = -1; // Reset, wenn der Pol wieder aufrecht ist
        }

        // 'truncated' wird in Gymnasium durch einen Wrapper (TimeLimit) gesetzt.
        // Hier ist es immer false, da wir den TimeLimit in der Hauptschleife behandeln.
        bool truncated_flag = false; 

        return {state, reward, terminated, truncated_flag};
    }

    // Hilfsfunktion zum Ausgeben des aktuellen Zustands
    void print_state() {
        std::cout << "x=" << state[0] << ", x_dot=" << state[1]
                  << ", theta=" << state[2] << ", theta_dot=" << state[3] << "\n";
    }
};

