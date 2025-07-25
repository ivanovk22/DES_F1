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

% Definition of parameters
m = 5; % m is the number of events
n = 18; % n is the number of states
ename = {'a1','a2','e1','e2','d'}; % original names of the events
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
F{1} = 'exprnd(1/lambda_1,1,L)'; % L values drawn from Exp(1/lambda)
F{2} = 'exprnd(1/lambda_2,1,L)'; % L values drawn from Exp(1/lambda)
F{3} = 'exprnd(1/mu_1,1,L)'; % L values drawn from Exp(1/mu)
F{4} = 'exprnd(1/mu_2,1,L)'; % L values drawn from Exp(1/mu)
F{5} = 'exprnd(1/sigma,1,L)'; % L values drawn from Exp(1/mu)

% MULTIPLE SIMULATIONS
disp('MULTIPLE SIMULATIONS'), disp(' ')

% Parameters of the simulations
kmax = 100; % maximum event index
M = 1000; % number of simulations(OUTER LOOP)
N = 10000; % number of experiments per simulation (INNER LOOP)

% Simulations


% Storage for batches results
nx_batches = zeros(M,n); % states count per batch (averaged over simulations)
% ne_batches = zeros(m, kmax, N);   % events count per batch (averaged over simulations)


disp(' Simulations in progress...')
for i = 1:M,
    % Progress
    if ismember(i,0:round(M/200):M),
        disp([ '   Progress ' num2str(i/M*100) '%' ])
    end
    EE = zeros(N,kmax);
    XX = zeros(N,kmax+1);
    TT = zeros(N,kmax+1);
    for sim_i = 1:N
        % Progress
        % if ismember(sim_i,0:round(N/200):N),
        %     disp([ '   Progress ' num2str(sim_i/N*100) '%' ])
        % end
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
        EE(sim_i,:) = E;
        XX(sim_i,:) = X;
        TT(sim_i,:) = T;
    end
        
       
    % Counting how many times the system is in each state at time tstar
    tol = 1e-10; % tolerance for time comparisons
    nx = zeros(1,n);
    r = (1:N)';
    c = sum(TT <= tstar+tol,2);
    ind = (c - 1) * N + r; % linear index
    for x = 1:n
        nx(1,x) = sum(XX(ind) == x);
    end
        
        % % Counting
        % for x = 1:n,
        %     nx(x,:) = nx(x,:)+(X(1:kmax+1)==x);
        % end
        % for e = 1:m,
        %     ne(e,:) = ne(e,:)+(E(1:kmax)==e);
        % end
    % Average over N simulations for this batch
    nx_batches(i, :) = nx / N;
    
end
% 
disp(' Simulations completed')
format short
% Compute mean and variance across M batches
px_mean = mean(nx_batches, 1) % [1 × n]
px_var  = var(nx_batches, 0, 1) % [1 × n]

% 
% % State probabilities: mean and variance (size: n x (kmax+1))
% px_mean = mean(nx_batches, 3);
% px_var  = var(nx_batches, 0, 3);
% steady_px_est = mean(px_mean(:, kmax-10:kmax), 2)
% % 
% % % Event probabilities: mean and variance (size: m x kmax)
% % pe_mean = mean(ne_batches, 3)
% % pe_var  = var(ne_batches, 0, 3);
% % figure;
% % for state = 1:n
% %     subplot(n,1,state)
% %     plot(0:kmax, px_mean(state,:), 'b', 'LineWidth', 1.5); hold on;
% %     plot(0:kmax, px_mean(state,:) + sqrt(px_var(state,:)), 'r--');
% %     plot(0:kmax, px_mean(state,:) - sqrt(px_var(state,:)), 'r--');
% %     title(['State ' num2str(state) ' Probability with ±1 std']);
% %     xlabel('Event index k'); ylabel('Probability');
% %     grid on;
% % end
% % 
% % 
% % % Print of the results
% % i = 1;
% % while 1,
% %     if (kmax/(10^(i-1)) < 10),
% %         break
% %     else
% %         i = i+1;
% %     end
% % end
% % 
% % disp(' ')
% % disp(' STATE PROBABILITIES (estimated)')
% % xcolumn = [];
% % for t = 1:kmax+1,
% %     j = 1;
% %     while 1,
% %         if ((t-1)/(10^(j-1)) < 10),
% %             break
% %         else
% %             j = j+1;
% %         end
% %     end
% %     xcolumn(t,:) = [ repmat(' ',1,i-j) ' X' num2str(t-1) ': ' ];
% % end
% % disp([ xcolumn num2str(px_mean') ])
% % 
% % disp(' ')
% % disp(' EVENT PROBABILITIES (estimated)')
% % ecolumn = [];
% % for t = 1:kmax,
% %     j = 1;
% %     while 1,
% %         if (t/(10^(j-1)) < 10),
% %             break
% %         else
% %             j = j+1;
% %         end
% %     end
% %     ecolumn(t,:) = [ repmat(' ',1,i-j) ' E' num2str(t) ': ' ];
% % end
% % disp([ ecolumn num2str(pe_mean') ])
% % disp(' ')
% % 
% % % Plots
% % figure, % state probabilities
% % str = 'plot(';
% % for x = 1:n,
% %     str = [ str '0:kmax,px_mean(' num2str(x) ',:),' ];
% % end
% % str = [ str(1:end-1) ')' ];
% % eval(str)
% % title('state probabilities')
% % xlabel('k')
% % ylabel('P(X_k = x)')
% % set(gca,'XTick',0:kmax,'XTickLabel',0:kmax)
% % xlim([ 0 kmax ])
% % str = 'legend(';
% % for x = 1:n,
% %     str = [ str '''P(X_k = ' xname{x} ')'',' ];
% % end
% % str = [ str(1:end-1) ')' ];
% % eval(str)
% % 
% % figure, % event probabilities
% % str = 'plot(';
% % for e = 1:m,
% %     str = [ str '1:kmax,pe_mean(' num2str(e) ',:),' ];
% % end
% % str = [ str(1:end-1) ')' ];
% % eval(str)
% % title('event probabilities')
% % xlabel('k')
% % ylabel('P(E_k = e)')
% % set(gca,'XTick',0:kmax,'XTickLabel',0:kmax)
% % xlim([ 1 kmax ])
% % str = 'legend(';
% % for e = 1:m,
% %     str = [ str '''P(E_k = ' ename{e} ')'',' ];
% % end
% % str = [ str(1:end-1) ')' ];
% % eval(str)
% % 
% % steady_px_est = mean(px_mean(:, kmax-10:kmax), 2)
% % 
