#include <fstream>
#include <iostream>

int main() {
    std::ofstream outFile("data/data.txt");

    if (!outFile.is_open()) {  // Prüft, ob die Datei geöffnet werden konnte
        std::cerr << "Fehler: Datei konnte nicht geöffnet werden!\n";
        return 1;  // Programm mit Fehlercode beenden
    }

    // Beispiel: einfache lineare Daten
    for (int i = 0; i < 1000000; ++i) {
        outFile << i << " " << i + i << "\n";  // x und y-Werte (z.B. x, x²)
    }    
    outFile.close();
    std::cout << "done!";
    return 0;
}
