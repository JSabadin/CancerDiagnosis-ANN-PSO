clear all; close all; clc;
file = open("data.mat")
data_matrix = file.data;% Load data matrix

% Separate data based on argument -- 357 benign, 212 malignant
malign_data = data_matrix(data_matrix(:,1)==2, :); % Data with argument 2
benign_data = data_matrix(data_matrix(:,1)==4, :); % Data with argument 4


% Set random seed for reproducibility
rng(1);

% Split data into train and test sets
train_ratio = 0.7; % Train ratio
malign_train_data = malign_data(randperm(size(malign_data,1), round(train_ratio*size(malign_data,1))), :);
benign_train_data = benign_data(randperm(size(benign_data,1), round(train_ratio*size(benign_data,1))), :);

train_data = [malign_train_data; benign_train_data];
test_data = setdiff(data_matrix, train_data, 'rows');

N_hidden = 5;    % Number of hidden neurons
N_features = 31; % The first is diagnosis

net = newff(train_data(:,2:end)', (train_data(:,1)' > 2), N_hidden,{'tansig','tansig'});
net = init(net);
net = configure(net,train_data(:,2:end)',(train_data(:,1)' > 2));


%  Weighted PSO (PSO-W: PSO with inertia weight) with mutations alghoritm
N_particles = 30;
N_iterations = 100;
c1 = 2.1; c2 = 2.1; w = 0.9; w_min = 0.4; w_max = 0.9; v_max = 0.2; mutation_rate = 0.05; mutation_scale = 0.1;

particle_positions = repmat(getwb(net)', N_particles, 1);
particle_velocities = zeros(size(particle_positions));
particle_personal_bests = particle_positions;
particle_personal_best_fitnesses = inf(N_particles, 1);
global_best = particle_positions(1,:);
global_best_fitness = inf;
train_acc = zeros(N_iterations,1);
test_acc = zeros(N_iterations,1);


for i = 1:N_iterations
    r1 = rand(size(particle_positions)); r2 = rand(size(particle_positions));
    particle_velocities = (w*particle_velocities + c1*r1.*(particle_personal_bests - particle_positions) + c2*r2.*(repmat(global_best,N_particles,1) - particle_positions));
    particle_velocities(particle_velocities > v_max) = v_max; particle_velocities(particle_velocities < -v_max) = -v_max;
    particle_positions = particle_positions + particle_velocities;
    for j = 1:N_particles
        if rand < mutation_rate
            r = randn(1,size(particle_positions,2));
            particle_positions(j,:) = particle_positions(j,:) + mutation_scale*(r/max(r));
        end
    end
    particle_positions(particle_positions > 1) = 1; particle_positions(particle_positions < -1) = -1;
    particle_fitnesses = zeros(N_particles,1);
    for j = 1:N_particles
        net = setwb(net, particle_positions(j,:)');
        outputs = sim(net, train_data(:,2:end)');
        errors = outputs - (train_data(:,1)' > 2);
        particle_fitnesses(j) = sum(errors.^2)/size(train_data,1);
        if particle_fitnesses(j) < particle_personal_best_fitnesses(j)
            particle_personal_bests(j,:) = particle_positions(j,:);
            particle_personal_best_fitnesses(j) = particle_fitnesses(j);
        end
        if particle_fitnesses(j) < global_best_fitness
            global_best = particle_positions(j,:);
            global_best_fitness = particle_fitnesses(j);
        end
    end
    w = max(w - (w_max - w_min)/N_iterations, w_min);
    train_acc(i) = sum(round(net(train_data(:,2:end)')) == (train_data(:,1)' > 2))/size(train_data,1);
    test_acc(i) = sum(round(net(test_data(:,2:end)')) == (test_data(:,1)' > 2))/size(test_data,1);
end

%% Analysing results

figure
plot(train_acc)
hold on;
plot(test_acc)
xlabel('Iterations');
ylabel('Accuracy');
legend('Train Accuracy', 'Test Accuracy', 'Location', 'south east');
fprintf("\nTrain accuracy is: %d \n",round(train_acc(end)*100));
fprintf("Test accuracy is: %d\n",round(test_acc(end)*100));

% (malign == 2,benign== 4) -> (malign == 0,benign== 1)
figure
plotconfusion( (test_data(:,1)' > 2), round(net(test_data(:,2:end)')))


view(net)
%% Linear activation function

net = newff(train_data(:,2:end)', (train_data(:,1)' > 2), N_hidden,{'purelin', 'purelin'});
net = init(net);
net = configure(net,train_data(:,2:end)',(train_data(:,1)' > 2));


%  Weighted PSO (PSO-W: PSO with inertia weight) with mutations alghoritm
N_particles = 40;
N_iterations = 100;
c1 = 2.1; c2 = 2.1; w = 0.9; w_min = 0.4; w_max = 0.9; v_max = 0.05; mutation_rate = 0.01; mutation_scale = 0.01;

particle_positions = repmat(getwb(net)', N_particles, 1);
particle_velocities = zeros(size(particle_positions));
particle_personal_bests = particle_positions;
particle_personal_best_fitnesses = inf(N_particles, 1);
global_best = particle_positions(1,:);
global_best_fitness = inf;
train_acc = zeros(N_iterations,1);
test_acc = zeros(N_iterations,1);


for i = 1:N_iterations
    r1 = rand(size(particle_positions)); r2 = rand(size(particle_positions));
    particle_velocities = (w*particle_velocities + c1*r1.*(particle_personal_bests - particle_positions) + c2*r2.*(repmat(global_best,N_particles,1) - particle_positions));
    particle_velocities(particle_velocities > v_max) = v_max; particle_velocities(particle_velocities < -v_max) = -v_max;
    particle_positions = particle_positions + particle_velocities;
    for j = 1:N_particles
        if rand < mutation_rate
            r = randn(1,size(particle_positions,2));
            particle_positions(j,:) = particle_positions(j,:) + mutation_scale*(r/max(r));
        end
    end
    particle_positions(particle_positions > 1) = 1; particle_positions(particle_positions < -1) = -1;
    particle_fitnesses = zeros(N_particles,1);
    for j = 1:N_particles
        net = setwb(net, particle_positions(j,:)');
        outputs = sim(net, train_data(:,2:end)');
        errors = outputs - (train_data(:,1)' > 2);
        particle_fitnesses(j) = sum(errors.^2)/size(train_data,1);
        if particle_fitnesses(j) < particle_personal_best_fitnesses(j)
            particle_personal_bests(j,:) = particle_positions(j,:);
            particle_personal_best_fitnesses(j) = particle_fitnesses(j);
        end
        if particle_fitnesses(j) < global_best_fitness
            global_best = particle_positions(j,:);
            global_best_fitness = particle_fitnesses(j);
        end
    end
    w = max(w - (w_max - w_min)/N_iterations, w_min);
    train_acc(i) = sum(round(net(train_data(:,2:end)')) == (train_data(:,1)' > 2))/size(train_data,1);
    test_acc(i) = sum(round(net(test_data(:,2:end)')) == (test_data(:,1)' > 2))/size(test_data,1);
    fprintf("Iteration %d\n",i);
end



% Using parameters as with different activation function we obtain "jumpy" results during training.
% Decrease the values of the acceleration coefficients c1 and c2. 
% These parameters control the influence of personal and global best positions on each particle's velocity,
% and decreasing them can make the algorithm more conservative and less prone to sudden changes.
% 
% Decrease the initial value of the inertia weight w. 
% This parameter controls the tendency of particles to maintain their current velocity and explore the search space. 
% Decreasing it can make the algorithm more cautious and less likely to overshoot optimal solutions.
% 
% Increase the value of the minimum inertia weight w_min.
% This parameter sets the lower bound for the inertia weight during the optimization process. 
% Increasing it can make the algorithm more stable and less susceptible to sudden changes.
% 
% Decrease the value of the maximum velocity v_max.
% This parameter limits the maximum change in particle positions during each iteration.
% Decreasing it can make the algorithm more conservative and less likely to jump to distant solutions.
% 
% Decrease the mutation rate mutation_rate or increase the mutation scale mutation_scale. 
% These parameters control the rate and magnitude of random mutations applied to particle positions during the optimization process.
% Changing them can make the algorithm more or less exploratory.
%% Analysing results for linear activation function

figure
plot(train_acc)
hold on;
plot(test_acc)
xlabel('Iterations');
ylabel('Accuracy');
legend('Train Accuracy', 'Test Accuracy', 'Location', 'south east');
fprintf("\nTrain accuracy is: %d \n",round(train_acc(end)*100));
fprintf("Test accuracy is: %d\n",round(test_acc(end)*100));


% Making errors at malign data and not at benign data (malign == 2,benign== 4)
figure
plotconfusion( (test_data(:,1)' > 2), round(net(test_data(:,2:end)')))


% malignant tumors are considered more serious because they have the potential to spread
% to other parts of the body and become life-threatening, while benign tumors are usually 
% less concerning and do not spread to other parts of the body.
