clear all % clears variables and functions from memory
close all % closes all open figures
clc % cleares the command window

% Parameters
% Poisson process with average 
% lap finish car 1 average
car1_lap_time = 1.5; %minutes
% lap finish car 2 average 
car2_lap_time = 1.5; %minutes
A1 = 1.4; B1 = 1.6; % minutes
% pit finish average
pit_time = 0.4; %minutes
A3 = 0.35; B3 = 0.45; % minutes
% need to pit for car 1 average
car1_pit_time = 15; %minutes
% need to pit for car 2 average
car2_pit_time = 15; %minutes
A2 = 14; B2 = 16; % minutes


p = 1/3; %probability of car 1 starting ahead
pi0 = [ p (1-p) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];

% Definition of parameters
m = 5; % m is the number of events
n = 18; % n is the number of states
ename = {'a1','a2','e1','e2','d'}; % original names of the events
xname = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'}; % original names of the states
tstar = 20; % time of interest

% Definition of the logical model
model.m = m;
model.n = n;

model.p = zeros(n,n,m); % transition probabilities
model.p(:,1,1) = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,1,2) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,1,3) = [0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,1,4) = [0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,1,5) = NaN(size(model.p, 1), 1);


model.p(:,2,1) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,2,2) = [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,2,3) = [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,2,4) = [0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,2,5) = NaN(size(model.p, 1), 1);

model.p(:,3,1) = NaN(size(model.p, 1), 1);
model.p(:,3,2) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,3,3) = NaN(size(model.p, 1), 1);
model.p(:,3,4) = [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,3,5) = NaN(size(model.p, 1), 1);

model.p(:,4,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0];
model.p(:,4,2) = [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,4,3) = NaN(size(model.p, 1), 1);
model.p(:,4,4) = NaN(size(model.p, 1), 1);
model.p(:,4,5) = NaN(size(model.p, 1), 1);

model.p(:,5,1) = [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,5,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0];
model.p(:,5,3) = NaN(size(model.p, 1), 1);
model.p(:,5,4) = NaN(size(model.p, 1), 1);
model.p(:,5,5) = NaN(size(model.p, 1), 1);

model.p(:,6,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0];
model.p(:,6,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
model.p(:,6,3) = NaN(size(model.p, 1), 1);
model.p(:,6,4) = NaN(size(model.p, 1), 1);
model.p(:,6,5) = NaN(size(model.p, 1), 1);

model.p(:,7,1) = [0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,7,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0];
model.p(:,7,3) = NaN(size(model.p, 1), 1);
model.p(:,7,4) = NaN(size(model.p, 1), 1);
model.p(:,7,5) = NaN(size(model.p, 1), 1);

model.p(:,8,1) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,8,2) = NaN(size(model.p, 1), 1);
model.p(:,8,3) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
model.p(:,8,4) = NaN(size(model.p, 1), 1);
model.p(:,8,5) = NaN(size(model.p, 1), 1);

model.p(:,9,1) = NaN(size(model.p, 1), 1);
model.p(:,9,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,9,3) = NaN(size(model.p, 1), 1);
model.p(:,9,4) = NaN(size(model.p, 1), 1);
model.p(:,9,5) = NaN(size(model.p, 1), 1);

model.p(:,10,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0];
model.p(:,10,2) = NaN(size(model.p, 1), 1);
model.p(:,10,3) = NaN(size(model.p, 1), 1);
model.p(:,10,4) = NaN(size(model.p, 1), 1);
model.p(:,10,5) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p(:,11,1) = NaN(size(model.p, 1), 1);
model.p(:,11,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0];
model.p(:,11,3) = NaN(size(model.p, 1), 1);
model.p(:,11,4) = NaN(size(model.p, 1), 1);
model.p(:,11,5) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p(:,12,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,12,2) = NaN(size(model.p, 1), 1);
model.p(:,12,3) = NaN(size(model.p, 1), 1);
model.p(:,12,4) = NaN(size(model.p, 1), 1);
model.p(:,12,5) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p(:,13,1) = NaN(size(model.p, 1), 1);
model.p(:,13,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1];
model.p(:,13,3) = NaN(size(model.p, 1), 1);
model.p(:,13,4) = NaN(size(model.p, 1), 1);
model.p(:,13,5) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p(:,14,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0];
model.p(:,14,2) = NaN(size(model.p, 1), 1);
model.p(:,14,3) = NaN(size(model.p, 1), 1);
model.p(:,14,4) = NaN(size(model.p, 1), 1);
model.p(:,14,5) = NaN(size(model.p, 1), 1);

model.p(:,15,1) = NaN(size(model.p, 1), 1);
model.p(:,15,2) = NaN(size(model.p, 1), 1);
model.p(:,15,3) = NaN(size(model.p, 1), 1);
model.p(:,15,4) = NaN(size(model.p, 1), 1);
model.p(:,15,5) = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p(:,16,1) = NaN(size(model.p, 1), 1);
model.p(:,16,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0];
model.p(:,16,3) = NaN(size(model.p, 1), 1);
model.p(:,16,4) = NaN(size(model.p, 1), 1);
model.p(:,16,5) = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p(:,17,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0];
model.p(:,17,2) = NaN(size(model.p, 1), 1);
model.p(:,17,3) = NaN(size(model.p, 1), 1);
model.p(:,17,4) = NaN(size(model.p, 1), 1);
model.p(:,17,5) = [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p(:,18,1) = NaN(size(model.p, 1), 1);
model.p(:,18,2) = NaN(size(model.p, 1), 1);
model.p(:,18,3) = NaN(size(model.p, 1), 1);
model.p(:,18,4) = NaN(size(model.p, 1), 1);
model.p(:,18,5) = [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p0 = pi0'; % initial state probabilities (vector n x 1)

% Definition of the stochastic clock structure
F{1} = 'unifrnd(A1,B1,1,L)'; % L values drawn from U(A1,B1)
F{2} = 'unifrnd(A1,B1,1,L)'; % L values drawn from U(A1,B1)
F{3} = 'unifrnd(A2,B2,1,L)'; % L values drawn from U(A2,B2)
F{4} = 'unifrnd(A2,B2,1,L)'; % L values drawn from U(A2,B2)
F{5} = 'unifrnd(A3,B3,1,L)'; % L values drawn from U(A3,B3)

% MULTIPLE SIMULATIONS
disp('MULTIPLE SIMULATIONS'), disp(' ')

% Parameters of the simulations
kmax = 100; % maximum event index
N = 1000000; % number of simulations

% Simulations
EE = zeros(N,kmax);
XX = zeros(N,kmax+1);
TT = zeros(N,kmax+1);
disp(' Simulations in progress...')
for i = 1:N,
    % Progress
    if ismember(i,0:round(N/200):N),
        disp([ '   Progress ' num2str(i/N*100) '%' ])
    end
    
    % Definition of the clock sequences
    L = kmax; % length of the clock sequences
    V = [];
    for j = 1:m,
        eval([ 'V(' num2str(j) ',:) = ' F{j} ';' ]);
    end
    
    % Simulation
    [E,X,T] = simprobdes(model,V);
    
    
    % Check
    if T(end) < tstar
        error('Insufficient number of events, increase ''kmax''')
    end
    
    % Store the simulation results
    EE(i,:) = E;
    XX(i,:) = X;
    TT(i,:) = T;
end
disp(' Simulations completed')

% Counting how many times the system is in each state at time tstar
tol = 1e-10; % tolerance for time comparisons
nx = zeros(1,n);
r = (1:N)';
c = sum(TT <= tstar+tol,2);
ind = (c - 1) * N + r; % linear index
for x = 1:n
    nx(1,x) = sum(XX(ind) == x);
end

% Estimating state probabilities at time tstar
px_est = nx/N
sum(px_est)

% error = abs(pi_tstar - px_est)


% Plot of state probabilities vs time
Tspan = 0:0.01:100; % grid of time values
L = length(Tspan);
nx = zeros(L,n);
r = (1:N)';
for j = 1:L
    c = sum(TT <= Tspan(j)+tol,2);
    ind = (c - 1) * N + r; % linear index
    for x = 1:n
        nx(j,x) = sum(XX(ind) == x); % counting
    end
end
PI_est = nx/N; % Estimating state probabilities 
figure
plot(Tspan,PI_est,tstar,px_est,'*')
title('Estimated state probabilities vs time')
xlabel('t [h]')
legend('\pi_1(t)','\pi_2(t)','\pi_3(t)','\pi_4(t)','\pi_5(t)',...
    '\pi_6(t)', '\pi_7(t)', '\pi_8(t)', '\pi_9(t)','\pi_1_0(t)',...
    '\pi_1_1(t)','\pi_1_2(t)','\pi_1_3(t)','\pi_1_4(t)','\pi_1_5(t)',...
    '\pi_1_6(t)','\pi_1_7(t)', '\pi_1_8(t)')