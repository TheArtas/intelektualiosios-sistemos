% 1. Duomenų generavimas
x = linspace(0, 1, 20);  % 20 skaičių intervale [0, 1]
y_desired = (1 + 0.6 * sin(2 * pi * x / 0.7) + 0.3 * sin(2 * pi * x)) / 2;  % Tikslinė funkcija

% 2. Rankiniu būdu parinkti centrų ir spindulių parametrai
c1 = 0.2; 
c2 = 0.9;
r1 = 0.15; 
r2 = 0.15;

% 3. Apskaičiuojamos Gauso funkcijų reikšmės kiekvienam įėjimo duomenų taškui
F1 = exp(-((x - c1).^2) / (2 * r1^2));
F2 = exp(-((x - c2).^2) / (2 * r2^2));

% 4. Sukuriama tinklo matricos reprezentacija
F = [F1; F2; ones(size(x))];  % Pridedama vienetų eilutė bias svoriui

% 5. Apskaičiuojami tinklo svoriai (w1, w2, w0) naudojant perceptrono mokymo algoritmą
W = pinv(F') * y_desired';  % Mažiausių kvadratų sprendinys (pseudo-inversija)

% 6. Apskaičiuojamas tinklo išėjimas ir klaida
y_approx = W' * F;
error = y_desired - y_approx;  % Klaida

% 7. Rezultatų vaizdavimas
figure;
plot(x, y_desired, 'r', 'LineWidth', 1.5);  % Tikslinė funkcija
hold on;
plot(x, y_approx, 'b--', 'LineWidth', 1.5);  % Aproksimuotas tinklo atsakas
legend('Tikslinė funkcija', 'SBF tinklo aproksimacija');
xlabel('x');
ylabel('y');
title('SBF Tinklo Aproksimacija');
grid on;
