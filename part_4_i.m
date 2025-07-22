clear all % clears variables and functions from memory
close all % closes all open figures
clc % cleares the command window

% Parameters
% Poisson process with average 
% lap finish car 1 average
car1_lap_time = 1.5; %minutes
% lap finish car 2 average 
car2_lap_time = 1.5; %minutes
% pit finish average
pit_time = 0.4 %minutes
% need to pit for car 1 average
car1_pit_time = 15 %minutes
% need to pit for car 2 average
car2_pit_time = 15 %minutes

lambda_1 = 1/car1_lap_time; % [laps/min]
lambda_2 = 1/car2_lap_time; % [laps/min]
sigma = 1/pit_time; % [finish/min]
mu_1 = 1/car1_pit_time; % [pit_needed/min]
mu_2 = 1/car2_pit_time; % [pit_needed/min]
p = 1/3; %probability of car 1 starting ahead
pi0 = [ p (1-p) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
tstar = 20; % time of interest

% Transition rate matrix
Q = [-(lambda_1+lambda_2+mu_1+mu_2)	lambda_2	lambda_1	mu_1	mu_2	0	0	0	0	0	0	0	0	0	0	0	0	0;
     lambda_1	-(lambda_1+lambda_2+mu_1+mu_2)	0	0	0	mu_1	mu_2	lambda_2	0	0	0	0	0	0	0	0	0	0;
     lambda_2	0	-(lambda_2+mu_2)	0	0	0	0	0	mu_2	0	0	0	0	0	0	0	0	0;
     0	0	0	-(lambda_1+lambda_2)	0	lambda_2	0	0	0	0	0	0	0	0	0	lambda_1	0	0;
     0	0	0	0	-(lambda_1+lambda_2)	0	0	0	lambda_1	0	0	lambda_2	0	0	0	0	0	0;
     0	0	0	0	0	-(lambda_1+lambda_2)	0	0	0	0	lambda_1	0	0	lambda_2	0	0	0	0;
     0	0	0	0	lambda_1	0	-(lambda_1+lambda_2)	0	0	0	0	0	0	0	0	0	lambda_2	0;
     0	lambda_1	0	0	0	0	0	-(lambda_1+mu_1)	0	0	0	0	0	mu_1	0	0	0	0;
     0	0	0	0	0	0	0	0	-lambda_2	lambda_2	0	0	0	0	0	0	0	0;
     sigma	0	0	0	0	0	0	0	0	-(sigma+lambda_1)	0	0	0	0	lambda_1	0	0	0;
     sigma	0	0	0	0	0	0	0	0	0	-(sigma+lambda_2)	0	lambda_2	0	0	0	0	0;
     0	sigma	0	0	0	0	0	0	0	lambda_1	0	-(sigma+lambda_1)	0	0	0	0	0	0;
     0	sigma	0	0	0	0	0	0	0	0	0	0	-(sigma+lambda_2)	0	0	0	0	lambda_2;
     0	0	0	0	0	0	0	0	0	0	0	0	lambda_1	-lambda_1	0	0	0	0;
     0	0	sigma	0	0	0	0	0	0	0	0	0	0	0	-sigma	0	0	0;
     0	0	sigma	0	0	0	0	0	0	0	lambda_2	0	0	0	0	-(sigma+lambda_2)	0	0;
     0	0	0	0	0	0	0	sigma	0	0	0	lambda_1	0	0	0	0	-(sigma+lambda_1)	0;
     0	0	0	0	0	0	0	sigma	0	0	0	0	0	0	0	0	0	-sigma];

% State probabilities at time tstar
pi_tstar = pi0*expm(Q*tstar)
sum(pi_tstar)
% Plot of state probabilities vs time
T = 0:0.01:30;
PI = [];
for t = T
    PI = [ PI ; pi0*expm(Q*t) ];
end
figure
plot(T,PI,tstar,pi_tstar,'*')
title('State probabilities vs time')
xlabel('t [min]')
legend('\pi_1(t)','\pi_2(t)','\pi_3(t)','\pi_4(t)','\pi_5(t)',...
    '\pi_6(t)', '\pi_7(t)', '\pi_8(t)', '\pi_9(t)','\pi_1_0(t)',...
    '\pi_1_1(t)','\pi_1_2(t)','\pi_1_3(t)','\pi_1_4(t)','\pi_1_5(t)',...
    '\pi_1_6(t)','\pi_1_7(t)', '\pi_1_8(t)')