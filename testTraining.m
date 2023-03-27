
%Get an environment Class.
testEnv = MistralRLEnvClass;
validateEnvironment(testEnv);

%Get a simple DQN agent to display training.

testActionInfo = rlFiniteSetSpec([1 2 3 4 5 6 7 8 9]);


testUavCoords = rlNumericSpec([1 3]);
testUserDistance = rlNumericSpec([1 2]);
testUserAngleHeading =  rlNumericSpec([1 2]);
testUserAnglePitch =    rlNumericSpec([1 2]);

testObsInfo(1) = testUavCoords;
testObsInfo(2) = testUserDistance;
testObsInfo(3) = testUserAngleHeading;
testObsInfo(4) = testUserAnglePitch;

%Agent generation simply for display purposes - I generally using learning
%rate optimizers etc. 

% Will try to show a fully converged agent at
%the end of the week.

%but for now I am having troubles with the use of the LOS_VECT function,

%If you look at the command window you can see a realtime display of
%several metrics.

%% Lester - Problems to look at. (please reach out if anything is unclear).
%   Line of sight to the users is represented by Logical values 1/0 at the
%   top, the LOS seems to always be [1 1] e.g the uav apparently can
%   always see the user in any scenario.

%  D1 Distance vect seems to be somewhat off, if you could have a quick
%  look to see if I'm using the LOS_VECT correclty.
testDQNAgent = rlDQNAgent(testObsInfo,testActionInfo);

%train agent
testTrainingStats = train(testDQNAgent,testEnv);
