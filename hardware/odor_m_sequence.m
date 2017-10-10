%% START TIMER FOR PULSE SEQUENCE PRESENTATION
tic;

%% DEFINE PARAMS
PULSE = 0.1;  % unit pulse duration (s)
ONSET = 90;  % start of pulse sequence (s)
OFFSET = 300;  % end of pulse sequence (s)
SEED = 0;  % integer RNG seed, which specifies the instantiation

M = 16;  % M memory elements
RCSN = [6, 4, 3, 1];  % recursion relatoin according to M (look up from 
% http://www.kempacoustics.com/thesis/node83.html#mlsrecurrsion)

TEST = 0;  % set to 1 to plot, instead of present, sequence
TEST_START = 88;  % start of test plot (s)
TEST_END = 95;  % end of test plot (s)

% round onset and offset to nearest pulse multiple
ONSET = round(ONSET/PULSE) * PULSE;
OFFSET = round(OFFSET/PULSE) * PULSE;

%% GENERATE M-SEQUENCE

rng(SEED);

% generate initial x whose first element is 1 (to ensure consistent
% first pulse times)
x = rand(1, M) < 0.5;
x(1) = 1;

% N total pulses
N = int64((OFFSET-ONSET)/PULSE) + 1;

% initialize sequence vector
s = zeros(N, 1);
s(1) = x(1);

% loop through all pulses
for ctr = 2:(N-1)
    % update x using M-sequence recursion relation
    x_end = mod(sum(x(RCSN)), 2);
    x = [x(2:end), x_end];
    % let first s equal first element of x
    s(ctr) = x(1);
end

if ONSET >= PULSE
    % pad s with zeros
    s = [zeros(int64(ONSET/PULSE), 1); s];
end

pulse_times = double(0:length(s)-1)' * PULSE;

disp('Sequence creation completed at t = ');
disp(toc);

%% PRESENT PULSE SEQUENCE
if ~TEST
    %% send to air tube

    for ctr = 1:length(pulse_times)
        next_pulse_time = pulse_times(ctr);

        % use "pause" if we'll have to wait a while for the next pulse
        if toc < (next_pulse_time - (0.9*PULSE))
            pause(0.9*PULSE);
        end
        
        % if for some reason we've missed this pulse skip it
        if toc > next_pulse_time
            if ctr == 1
                s(ctr) = 0;
            else
                s(ctr) = s(ctr-1);
            end
        end
        
        % hold till next pulse time
        while toc < next_pulse_time
            continue
        end

        % send signal
        if s(ctr) == 0
            % send odor off signal
            disp('OFF');
        elseif s(ctr) == 1
            % send odor on signal
            disp('ON');
        end

    end
else
    %% send to plot
    
    % make time vector to display
    t = [pulse_times'; [pulse_times(2:end); OFFSET+PULSE]'];
    t_ = t(:);
    
    % make odor vector to display
    s_ = [s'; s'];
    s_ = s_(:);
    
    plot(t_, s_);
    xlim([TEST_START, TEST_END]);
    xlabel('time (s)');
    ylabel('odor state');
    
end