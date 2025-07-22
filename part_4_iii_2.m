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
NEW_RATE = 1/3;
p = 1/3; %probability of car 1 starting ahead
pi0 = [ p (1-p) 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
% tstar = 2; % time of interest
% 
% 
% % Transition rate matrix
% Q = [-(lambda_1+lambda_2+mu_1+mu_2)	lambda_2	lambda_1	mu_1	mu_2	0	0	0	0	0	0	0	0	0	0	0	0	0;
%      lambda_1	-(lambda_1+lambda_2+mu_1+mu_2)	0	0	0	mu_1	mu_2	lambda_2	0	0	0	0	0	0	0	0	0	0;
%      lambda_2	0	-(lambda_2+mu_2)	0	0	0	0	0	mu_2	0	0	0	0	0	0	0	0	0;
%      0	0	0	-(lambda_1+lambda_2)	0	lambda_2	0	0	0	0	0	0	0	0	0	lambda_1	0	0;
%      0	0	0	0	-(lambda_1+lambda_2)	0	0	0	lambda_1	0	0	lambda_2	0	0	0	0	0	0;
%      0	0	0	0	0	-(lambda_1+lambda_2)	0	0	0	0	lambda_1	0	0	lambda_2	0	0	0	0;
%      0	0	0	0	lambda_1	0	-(lambda_1+lambda_2)	0	0	0	0	0	0	0	0	0	lambda_2	0;
%      0	lambda_1	0	0	0	0	0	-(lambda_1+mu_1)	0	0	0	0	0	mu_1	0	0	0	0;
%      0	0	0	0	0	0	0	0	-lambda_2	lambda_2	0	0	0	0	0	0	0	0;
%      sigma	0	0	0	0	0	0	0	0	-(sigma+lambda_1)	0	0	0	0	lambda_1	0	0	0;
%      sigma	0	0	0	0	0	0	0	0	0	-(sigma+lambda_2)	0	lambda_2	0	0	0	0	0;
%      0	sigma	0	0	0	0	0	0	0	lambda_1	0	-(sigma+lambda_1)	0	0	0	0	0	0;
%      0	sigma	0	0	0	0	0	0	0	0	0	0	-(sigma+lambda_2)	0	0	0	0	lambda_2;
%      0	0	0	0	0	0	0	0	0	0	0	0	lambda_1	-lambda_1	0	0	0	0;
%      0	0	sigma	0	0	0	0	0	0	0	0	0	0	0	-sigma	0	0	0;
%      0	0	sigma	0	0	0	0	0	0	0	lambda_2	0	0	0	0	-(sigma+lambda_2)	0	0;
%      0	0	0	0	0	0	0	sigma	0	0	0	lambda_1	0	0	0	0	-(sigma+lambda_1)	0;
%      0	0	0	0	0	0	0	sigma	0	0	0	0	0	0	0	0	0	-sigma];
% 
% % State probabilities at time tstar
% pi_tstar = pi0*expm(Q*tstar)
% sum(pi_tstar)

% Definition of parameters
m = 6; % m is the number of events
n = 18; % n is the number of states
ename = {'a1','a2','e1','e2','d','NEW'}; % original names of the events
xname = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'}; % original names of the states

% Definition of the logical model
model.m = m;
model.n = n;

model.p = zeros(n,n,m); % transition probabilities
model.p(:,1,1) = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,1,2) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,1,3) = [0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,1,4) = [0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,1,5) = NaN(size(model.p, 1), 1);
model.p(:,1,6) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

model.p(:,2,1) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,2,2) = [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,2,3) = [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,2,4) = [0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,2,5) = NaN(size(model.p, 1), 1);
model.p(:,2,6) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,3,1) = NaN(size(model.p, 1), 1);
model.p(:,3,2) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,3,3) = NaN(size(model.p, 1), 1);
model.p(:,3,4) = [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,3,5) = NaN(size(model.p, 1), 1);
model.p(:,3,6) = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,4,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0];
model.p(:,4,2) = [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,4,3) = NaN(size(model.p, 1), 1);
model.p(:,4,4) = NaN(size(model.p, 1), 1);
model.p(:,4,5) = NaN(size(model.p, 1), 1);
model.p(:,4,6) = [0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,5,1) = [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,5,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0];
model.p(:,5,3) = NaN(size(model.p, 1), 1);
model.p(:,5,4) = NaN(size(model.p, 1), 1);
model.p(:,5,5) = NaN(size(model.p, 1), 1);
model.p(:,5,6) = [0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,6,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0];
model.p(:,6,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
model.p(:,6,3) = NaN(size(model.p, 1), 1);
model.p(:,6,4) = NaN(size(model.p, 1), 1);
model.p(:,6,5) = NaN(size(model.p, 1), 1);
model.p(:,6,6) = [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,7,1) = [0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,7,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0];
model.p(:,7,3) = NaN(size(model.p, 1), 1);
model.p(:,7,4) = NaN(size(model.p, 1), 1);
model.p(:,7,5) = NaN(size(model.p, 1), 1);
model.p(:,7,6) = [0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,8,1) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,8,2) = NaN(size(model.p, 1), 1);
model.p(:,8,3) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
model.p(:,8,4) = NaN(size(model.p, 1), 1);
model.p(:,8,5) = NaN(size(model.p, 1), 1);
model.p(:,8,6) = [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,9,1) = NaN(size(model.p, 1), 1);
model.p(:,9,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,9,3) = NaN(size(model.p, 1), 1);
model.p(:,9,4) = NaN(size(model.p, 1), 1);
model.p(:,9,5) = NaN(size(model.p, 1), 1);
model.p(:,9,6) = [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,10,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0];
model.p(:,10,2) = NaN(size(model.p, 1), 1);
model.p(:,10,3) = NaN(size(model.p, 1), 1);
model.p(:,10,4) = NaN(size(model.p, 1), 1);
model.p(:,10,5) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,10,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];


model.p(:,11,1) = NaN(size(model.p, 1), 1);
model.p(:,11,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0];
model.p(:,11,3) = NaN(size(model.p, 1), 1);
model.p(:,11,4) = NaN(size(model.p, 1), 1);
model.p(:,11,5) = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,11,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0];


model.p(:,12,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,12,2) = NaN(size(model.p, 1), 1);
model.p(:,12,3) = NaN(size(model.p, 1), 1);
model.p(:,12,4) = NaN(size(model.p, 1), 1);
model.p(:,12,5) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,12,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0];

model.p(:,13,1) = NaN(size(model.p, 1), 1);
model.p(:,13,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1];
model.p(:,13,3) = NaN(size(model.p, 1), 1);
model.p(:,13,4) = NaN(size(model.p, 1), 1);
model.p(:,13,5) = [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,13,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0];

model.p(:,14,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0];
model.p(:,14,2) = NaN(size(model.p, 1), 1);
model.p(:,14,3) = NaN(size(model.p, 1), 1);
model.p(:,14,4) = NaN(size(model.p, 1), 1);
model.p(:,14,5) = NaN(size(model.p, 1), 1);
model.p(:,14,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];


model.p(:,15,1) = NaN(size(model.p, 1), 1);
model.p(:,15,2) = NaN(size(model.p, 1), 1);
model.p(:,15,3) = NaN(size(model.p, 1), 1);
model.p(:,15,4) = NaN(size(model.p, 1), 1);
model.p(:,15,5) = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,15,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0];


model.p(:,16,1) = NaN(size(model.p, 1), 1);
model.p(:,16,2) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0];
model.p(:,16,3) = NaN(size(model.p, 1), 1);
model.p(:,16,4) = NaN(size(model.p, 1), 1);
model.p(:,16,5) = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,16,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0];


model.p(:,17,1) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0];
model.p(:,17,2) = NaN(size(model.p, 1), 1);
model.p(:,17,3) = NaN(size(model.p, 1), 1);
model.p(:,17,4) = NaN(size(model.p, 1), 1);
model.p(:,17,5) = [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,17,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0];


model.p(:,18,1) = NaN(size(model.p, 1), 1);
model.p(:,18,2) = NaN(size(model.p, 1), 1);
model.p(:,18,3) = NaN(size(model.p, 1), 1);
model.p(:,18,4) = NaN(size(model.p, 1), 1);
model.p(:,18,5) = [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
model.p(:,18,6) = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1];


model.p0 = pi0'; % initial state probabilities (vector n x 1)

% Definition of the stochastic clock structure
F{1} = 'exprnd(1/lambda_1,1,L)'; % L values drawn from Exp(1/lambda)
F{2} = 'exprnd(1/lambda_2,1,L)'; % L values drawn from Exp(1/lambda)
F{3} = 'exprnd(1/mu_1,1,L)'; % L values drawn from Exp(1/mu)
F{4} = 'exprnd(1/mu_2,1,L)'; % L values drawn from Exp(1/mu)
F{5} = 'exprnd(1/sigma,1,L)'; % L values drawn from Exp(1/mu)
F{6} = 'exprnd(1/NEW_RATE,1,L)'; % L values drawn from Exp(1/NEW)


% MULTIPLE SIMULATIONS
disp('MULTIPLE SIMULATIONS'), disp(' ')

% Parameters of the simulations
kmax = 1000; % maximum event index
N = 10000; % number of simulations

% Simulations
EE = zeros(N,kmax);
XX = zeros(N,kmax+1);
TT = zeros(N,kmax+1);
disp(' Simulations in progress...')


avg_lap_times = zeros(1, N);
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
    % if T(end) < tstar
    %     error('Insufficient number of events, increase ''kmax''')
    % end
    
    

    % Store the simulation results
    EE(i,:) = E;
    XX(i,:) = X;
    TT(i,:) = T;

    %car1_laps = sum(E(T(1:kmax) <= tstar) == 1); %for time before
                                                    %tstar

   % car1_laps = diff(T(E == 5)); %interevent times for a1
    %avg_vals2(i) = mean(car1_laps);
    
    a1_event_indices = find(E == 1);
    a1_timestamps = T(a1_event_indices + 1);
    measured_lap_times_a1 = diff(a1_timestamps);
    avg_lap_times_a1(i) = mean(measured_lap_times_a1);
    avg_lap_times_var_a1(i) = var(measured_lap_times_a1);

    a2_event_indices = find(E == 2);
    a2_timestamps = T(a2_event_indices + 1);
    measured_lap_times_a2 = diff(a2_timestamps);
    avg_lap_times_a2(i) = mean(measured_lap_times_a2);
    avg_lap_times_var_a2(i) = var(measured_lap_times_a2);

    a3_event_indices = find(E == 3);
    a3_timestamps = T(a3_event_indices + 1);
    measured_lap_times_a3 = diff(a3_timestamps);
    avg_lap_times_a3(i) = mean(measured_lap_times_a3);
    avg_lap_times_var_a3(i) = var(measured_lap_times_a3);

    a4_event_indices = find(E == 4);
    a4_timestamps = T(a4_event_indices + 1);
    measured_lap_times_a4 = diff(a4_timestamps);
    avg_lap_times_a4(i) = mean(measured_lap_times_a4);
    avg_lap_times_var_a4(i) = var(measured_lap_times_a4);

    a5_event_indices = find(E == 5);
    a5_timestamps = T(a5_event_indices + 1);
    measured_lap_times_a5 = diff(a5_timestamps);
    avg_lap_times_a5(i) = mean(measured_lap_times_a5);
    avg_lap_times_var_a5(i) = var(measured_lap_times_a5);

    a6_event_indices = find(E == 6);
    a6_timestamps = T(a6_event_indices + 1);
    measured_lap_times_a6 = diff(a6_timestamps);
    avg_lap_times_a6(i) = mean(measured_lap_times_a6);
    avg_lap_times_var_a6(i) = var(measured_lap_times_a6);




end
disp(' Simulations completed')  
avg_a1 = mean(avg_lap_times_a1)
avg_a1_var = mean(avg_lap_times_var_a1)

avg_a2 = mean(avg_lap_times_a2)
avg_a2_var = mean(avg_lap_times_var_a2)

avg_a3 = mean(avg_lap_times_a3)
avg_a3_var = mean(avg_lap_times_var_a3)

avg_a4 = mean(avg_lap_times_a4)
avg_a4_var = mean(avg_lap_times_var_a4)

avg_a5 = mean(avg_lap_times_a5)
avg_a5_var = mean(avg_lap_times_var_a5)

avg_a6 = mean(avg_lap_times_a6)
avg_a6_var = mean(avg_lap_times_var_a6)

%idxOnes = find(E == 5);
%idxBefore = idxOnes - 1;
%TT_d = TT(:,idxOnes) - TT(:,idxBefore);

%avg_fin = mean(TT_d,2);
%avg_fin = mean(avg_fin, 1)


%     % Counting how many times the system is in each state at time tstar
%     tol = 1e-10; % tolerance for time comparisons
%     nx = zeros(1,n);
%     r = (1:N)';
%     c = sum(TT <= tstar+tol,2);
%     ind = (c - 1) * N + r; % linear index
%     for x = 1:n
%         nx(1,x) = sum(XX(ind) == x);
%     end
% 
% % Estimating state probabilities at time tstar
%     px_est = nx/N

 



%final_avg = mean(avg_vals(1))
%final_variance = var(avg_vals)