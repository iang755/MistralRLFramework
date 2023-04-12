function simResults = simAgent(numUsers)
load("agent2Long_Trained.mat");

simOptions = rlSimulationOptions();
simOptions.MaxSteps = 150;
simOptions.NumSimulations = 20;

%% Perform simulations
simResults = sim(MistralRLEnvClass(true,numUsers),agent2Long_Trained,simOptions);
