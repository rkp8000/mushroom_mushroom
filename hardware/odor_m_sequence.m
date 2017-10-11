%% START TIMER FOR PULSE SEQUENCE PRESENTATION
tic;

%% DEFINE PARAMS
SAVE_FILE_PREFIX = 'odor_times';

% RNG SEEDS
SEED_P = 'shuffle';
SEED_M = 'shuffle';

% PULSE/ITVL SEQUENCE
FIRST_PULSE = 30;  % time of first pulse (s)
P_DURS = [1, 5];  % range of pulse duration (s)
I_DURS = [10, 20];  % range of interpulse interval duration (s)
P_OFFSET = 230;  % time to stop pulse/interval sequence (s)

% M-SEQUENCE
M_ONSET = 240;  % start of M-sequence (s) (must be after P_OFFSET)
M_OFFSET = 300;  % end of M-sequence (s)
M_UNIT = 0.5;  % unit pulse/no-pulse interval in M-sequence (s)

M = 16;  % M memory elements
RCSN = [6, 4, 3, 1];  % recursion relation according to M (look up from 
% http://www.kempacoustics.com/thesis/node83.html#mlsrecurrsion)

% TEST/VIEWING PARAMS
TEST = 1;  % set to 1 to plot, instead of present, sequence
TEST_START = 0;  % start of test plot (s)
TEST_END = 300;  % end of test plot (s)

if P_OFFSET > M_ONSET
    error('P_OFFSET MUST BE BEFORE M_ONSET');
end

%% MAKE PULSE/ITVL SEQUENCE
rng(SEED_P);

max_pulses = ceil((P_OFFSET - FIRST_PULSE) / (P_DURS(1) + I_DURS(1)));

odor_on_times_p = nan(max_pulses, 1);
odor_off_times_p = nan(max_pulses, 1);

t = FIRST_PULSE;
ctr = 0;

while t < P_OFFSET
    
    ctr = ctr + 1;
    
    % pulse
    odor_on_times_p(ctr) = t;
    pulse_dur = P_DURS(1) + ((P_DURS(2) - P_DURS(1))*rand());
    t = t + pulse_dur;
    odor_off_times_p(ctr) = t;
    
    % interval
    itvl_dur = I_DURS(1) + ((I_DURS(2) - I_DURS(1))*rand());
    t = t + itvl_dur;
    
end

odor_on_times_p = odor_on_times_p(1:ctr, 1);
odor_off_times_p = odor_off_times_p(1:ctr, 1);

% make sure last pulse does not extend beyond P_OFFSET
odor_off_times_p(end) = min(odor_off_times_p(end), P_OFFSET);

%% MAKE M-SEQUENCE
rng(SEED_M);

M_DUR = M_OFFSET - M_ONSET;

% N total time units for M-seq
N = int64(floor(M_DUR/M_UNIT)) + 1;

% initialize final result vector
s = zeros(N, 1);
s(1) = 1;

% generate initial x whose first element is 1 (to ensure consistent
% first pulse times)
x = rand(1, M) < 0.5;
x(1) = 1;

% loop through all time units
for ctr = 2:(N-1)
    % update x using M-seq recursion relation
    x_end = mod(sum(x(RCSN)), 2);
    x = [x(2:end), x_end];
    % let first s equal first element of x
    s(ctr) = x(1);
end

% extract odor on and off times
odor_on_times_m = M_ONSET + (M_UNIT * (find(diff([0; s]) == 1) - 1));
odor_off_times_m = M_ONSET + (M_UNIT * (find(diff([0; s]) == -1) - 1));

%% COMBINE INTO FULL SEQUENCES OF ODOR ON/OFF TIMES

odor_on_times = [odor_on_times_p; odor_on_times_m];
odor_off_times = [odor_off_times_p; odor_off_times_m];

%% SAVE ODOR ON AND OFF TIMES TO FILE
timestamp = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
save_file = [SAVE_FILE_PREFIX, '_', timestamp, '.mat'];
save(save_file, 'odor_on_times_p', 'odor_off_times_p', 'odor_on_times_m', 'odor_off_times_m');

%% PRESENT PULSE SEQUENCE

if ~TEST
    %% send to air tube
    for pulse_ctr = 1:length(odor_on_times)
        
        % get on and off times of next pulse
        odor_on_time = odor_on_times(pulse_ctr);
        odor_off_time = odor_off_times(pulse_ctr);
        
        t = toc;
        
        if odor_off_time <= t
            % continue if we've completely missed this pulse
            continue
        elseif odor_on_time <= t
            % turn odor on if we've missed the start of the pulse
            disp('ON'); disp(toc);
            
            % pause up to 0.05 seconds before pulse end
            till_odor_off = odor_off_time - toc;
            
            if till_odor_off > 0.05
                pause(till_odor_off - 0.05);
            end
            
            % wait till correct moment to turn odor off
            while toc < odor_off_time
                continue
            end
            
            % turn odor off
            disp('OFF'); disp(toc);
            
        else
            % pause up to 0.05 seconds before pulse start
            till_odor_on = odor_on_time - toc;
            
            if till_odor_on > 0.05
                pause(till_odor_on - 0.05);
            end
            
            % wait till correct moment to turn odor on
            while toc < odor_on_time
                continue
            end
            
            % turn odor on
            disp('ON'); disp(toc);
            
            % pause up to 0.05 seconds before pulse end
            till_odor_off = odor_off_time - toc;
            
            if till_odor_off > 0.05
                pause(till_odor_off - 0.05);
            end
            
            % wait till correct moment to turn odor off
            while toc < odor_off_time
                continue
            end
            
            % turn odor off
            disp('OFF'); disp(toc);
            
        end
    end
else
    %% send to plot
    % make time vector
    temp_t = [odor_on_times, odor_off_times]';
    temp_t = [temp_t(:), temp_t(:)]';
    ts = [0; temp_t(:); M_OFFSET];
    
    % make odor vector
    temp_o = repmat([0, 1, 1, 0]', 1, length(odor_on_times));
    os = [0; temp_o(:); 0];
    
    % display plot
    
    plot(ts, os);
    xlim([TEST_START, TEST_END]);
    xlabel('time (s)');
    ylabel('odor state');
    
end