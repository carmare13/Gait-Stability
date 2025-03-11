
%function [spatiotemporal_variables] = GaitPrint_Spatiotemporal_Calculation(base_filename,...
    %dat, sampling_rate, png_filename, png, plot_option, save_option)

% INPUTS:
% All inputs should be the raw values from the exported NORAXON data.

% dat = Datafile being manipulated sampling_rate = Sampling rate used to
% capture data png_filename = Full filename being used png = Full filename
% being used + ".png" plot_option = Figure popups (1 for yes, 0 for no)
% save_option = Save figure (1 for yes, 0 for no). If save_option is 1 for
% yes make sure to change the path location on line 534.

% OUTPUT:

% Single table with the following columns:
% cadence = Number of steps per minute.
% step_time = Time elapsed between initial contacts of opposite feet.
% left/right_step_length = Distance covered in centimeters between initial
% contacts of opposite feet.
% left/right_step_width = Lateral distance between the heel center of one
% footprint and the line joining the heel center of two consecutive
% footprints of the opposite foot.
% left/right_stride_length = Distance covered in centimeters from
% consecutive heel strikes of the same foot.
% left/right_stride_time = Time elapsed between two consecutive footfalls
% of the same foot.
% left/right_stance_time = Time of gait cycle when the foot is in contact
% with the ground between strides.
% left/right_swing_time = Time of gait cycle when the foot is not in
% contact with the ground between strides.
% single_support_time = Time of gait cycle when only one foot is in contact
% with the ground.
% double_support_time = Time of gait cycle when both feet are in contact
% with the ground.
% left/right_pct_stance = Stance time normalized to stride time.
% left/right_pct_swing = Swing time normalized to stride time.
% pct_single = Single support time normalized to stride time.
% pct_double = Double support time normalized to stride time.
% average_speed = Average speed in meters per second.
% left/right_stride_speed = Speed of the limb from toe off to heel strike.
% distance_traveled = Distance traveled in meters.

% Stride distance is calculated using the X and Y
% trajectories at each heel strike as the coordinates to use in the
% Euclidean Distance equation.

% Step width is calculated by finding all side lengths in the below
% triangle. Once the triangle is created and all internal angles found, the
% angle at L1 is divided by two. The assumption is made here that the
% resulting line drawn across intercepts the R1-R2 line at 90 degrees. The
% length of this line is the step width.

% Finally step length is calculated by finding the length of the side
% opposite the hypotenuse of a right angle triangle.

% Example for calculating spatiotemporal parameters for the left leg
%         ^                           R2       ___
%        /|\                          /|        |
%         |    Direction             / |        |
%         |      of                 /  |        |
%         |    Travel              /   |        |
%                                 /    |        | Stride
%                          ___ L1 -----| Step   | Length
%                           |     \    | Width  |
%                           |      \   |        |
%                 Step      |       \  |        |
%                 Length    |        \ |        |
%                          _|_        \|       _|_
%                                     R1

%%
% TITLE: GaitPrint_Spatiotemporal_Calculation.m
% DATE: September 1, 2022
% AUTHOR: Tyler M. Wiles, MS
% EMAIL: tylerwiles@unomaha.edu

% DESCRIPTION:
% This code will calculate spatiotemporal gait parameters from Noraxon
% Ultium IMU data based on the foot contact values calculated by the MR3.18
% program itself.

% Copyright 2022, Tyler M. Wiles, Joel H. Sommerfeld.

% Redistribution and use of this script, with or without
% modification, is permitted provided this copyright notice,
% the original authors name and the following disclaimer remains.

% DISCLAIMER: It is the user's responsibility to check the code is returning
% the appropriate results before reporting any scientific findings.
% The author will not accept any responsibility for incorrect results
% directly relating to user error.

% NOTE: This code relies on the use of the function padcat.m. This is not a
% built-in MATLAB function and can be downloaded from MathWorks. This
% file needs to be in your directory path to run.
% <https://www.mathworks.com/matlabcentral/fileexchange/22909-padcat>

dat= readmatrix('S001_G01_D01_B01_T01.csv');
sampling_rate =200; %Noraxon Ultium Motion inertial measurement unit
plot_option = 0;
dar=dat(:,245);
dal=dat(:,154); 
%% Figure Handling

% Turning figure popups on/off. big_figure is an empty figure template to
% be used later for our big subplot.
% if plot_option == 1
%     set(groot,'defaultFigureVisible','on')
%     big_figure = figure;
% else
%     set(groot,'defaultFigureVisible','off')
%     big_figure = figure;
% end

%% Find Foot Contact Events

% Finding right heel contact locations
[~, right_contact_locs] = findpeaks(dat(:,245),'MinPeakWidth',70, 'MinPeakDistance',50,'MinPeakHeight',0.5);

% Finding right toe off locations
[~, right_toeoff_locs] = findpeaks(-dat(:,245),'MinPeakWidth',50);

% Finding left heel contact locations
[~, left_contact_locs] = findpeaks(dat(:,154),'MinPeakWidth',70, 'MinPeakDistance',50, 'MinPeakHeight',0.5);

% Finding left toe off locations
[~, left_toeoff_locs] = findpeaks(-dat(:,154),'MinPeakWidth',50);

%% If statements to fix trial specific issues

% if base_filename == 'S015_G01_D02_B01_T02.csv'
%     left_contact_locs(87) = []; % Remove false heel contact
% end
% 
% % Missing heel contacts are synthesized by adding average amount of frames
% % between contacts to the first contact prior to the missing contact.
% if base_filename == 'S016_G01_D01_B01_T01.csv'
%     top = left_contact_locs(1:52); bot = left_contact_locs(53:end);
%     left_contact_locs = [top; 10814; bot];
% end
% 
% if base_filename == 'S016_G01_D01_B03_T03.csv'
%     top = left_contact_locs(1:64); mid = left_contact_locs(65:74); bot = left_contact_locs(75:end);
%     left_contact_locs = [top; 13417; mid; 15668; bot]; % Fill missing contact 1
%     top = left_toeoff_locs(1:65); mid = left_toeoff_locs(66:75); bot = left_toeoff_locs(76:end);
%     left_toeoff_locs = [top; 13545; mid; 15796; bot]; % Fill missing contact 2
% end
% 
% if base_filename == 'S048_G02_D01_B02_T03.csv'
%     right_contact_locs(1) = []; % First contact is false
% end

%% If statement to only do last step for faulty heel contact trials

% If statement in case the heel contacts are not properly registering. In
% the case of S039 or S012 not all peaks occur for example. In some cases
% (S063) there are so many missing contacts (defined at 20) that we cannot
% accurately synthesize heel contacts similarly to the section above,
% therefore these calculations are entered as NaNs.
if size(left_contact_locs) <= 10 | size(right_contact_locs) <= 10 |...
        abs(length(left_contact_locs) - length(right_contact_locs)) > 20
    
    % Create empty array since we are not calculating spatiotemporal variables
    % due to heel contacts not registering
    spatiotemporal_variables = NaN(1,26,'single');
    
    % Change to table and name variables for later use
    spatiotemporal_variables = array2table(spatiotemporal_variables);
    spatiotemporal_variables.Properties.VariableNames = {'cadence (steps/min)',...
        'step time (s)', 'left step length (cm)', 'right step length (cm)',...
        'left step width (cm)', 'right step width (cm)', 'left stride length (cm)',...
        'right stride length (cm)', 'left stride time (s)','right stride time (s)',...
        'left stance time (s)', 'right stance time (s)','left swing time (s)',...
        'right swing time (s)', 'single support_time (s)','double support time (s)',...
        'left pct stance (%GC)', 'right pct stance (%GC)','left pct swing (%GC)',...
        'right pct swing (%GC)', 'pct single (%GC)','pct double (%GC)',...
        'average speed (m/s)', 'left stride speed (m/s)', 'right stride speed (m/s)',...
        'distance traveled (m)'};
    
else
    
    %% Spatial Calculations
    
    leftXTraj = dat(:,318);
    leftYTraj = dat(:,319);
    rightXTraj = dat(:,320);
    rightYTraj = dat(:,321);
    
    % For the left leg
    for i = 1:length(right_contact_locs)-1
        
        % Get the foot locations
        R1x = rightXTraj(right_contact_locs(i,1),1);
        R1y = rightYTraj(right_contact_locs(i,1),1);
        
        R2x = rightXTraj(right_contact_locs(i+1,1),1);
        R2y = rightYTraj(right_contact_locs(i+1,1),1);
        
        L1x = leftXTraj(left_contact_locs(i,1),1);
        L1y = leftYTraj(left_contact_locs(i,1),1);
        
        % Calculate distance from right one to right two --> stride length
        right_stride = sqrt((R2x - R1x)^2 + (R2y - R1y)^2);
        c = right_stride/10;
        right_stride_length(i,1) = right_stride/10;
        
        % Calculate distance from right one to left one
        R1L1 = sqrt((L1x - R1x)^2 + (L1y - R1y)^2);
        b = R1L1/10;
        
        % Calculate distance from right two to left one
        R2L1 = sqrt((L1x - R2x)^2 + (L1y - R2y)^2);
        a = R2L1/10;
        
        % Calculate left step width
        semi_perimeter = (a+b+c)/2;

        left_step_width(i,1) = (2*sqrt(semi_perimeter*(semi_perimeter - a)*(semi_perimeter - b)*(semi_perimeter - c)))/b;
        
        % Calculate left step length
        left_step_length(i,1) = real(sqrt((b^2) - (left_step_width(i,1)^2)));
    end
    
    % For the right leg
    for i = 1:length(left_contact_locs)-1
        
%         % Adding the foot trajectory plot to the big subplot
%         % Has to be done here because this is where the stride/step
%         % calculations take place.
%         set(0, 'CurrentFigure', big_figure);
%         subplot(4,3,[2;5])
%         title('Foot Placement');
%         hold on
        
        % Get the foot locations
        L1x = rightXTraj(left_contact_locs(i,1),1);
        L1y = rightYTraj(left_contact_locs(i,1),1);
        
        L2x = rightXTraj(left_contact_locs(i+1,1),1);
        L2y = rightYTraj(left_contact_locs(i+1,1),1);
        
        R1x = leftXTraj(right_contact_locs(i,1),1);
        R1y = leftYTraj(right_contact_locs(i,1),1);
        
        plot(L1x, L1y, "Marker","pentagram", "Color", "k", "MarkerSize", 6)
        plot(L2x, L2y, "Marker","hexagram", "Color", "b", "MarkerSize", 4)
        plot(R1x, R1y, "Marker","o", "Color", "g", "MarkerSize", 3)
        
        P = [L1x, L1y; L2x, L2y; R1x, R1y];
        T = delaunayTriangulation(P);
        triplot(T)
        
        % Calculate distance from left one to left two --> stride length
        left_stride = sqrt((L2x - L1x)^2 + (L2y - L1y)^2);
        c = left_stride/10;
        left_stride_length(i,1) = left_stride/10;
        
        % Calculate distance from left one to right one
        L1R1 = sqrt((R1x - L1x)^2 + (R1y - L1y)^2);
        b = L1R1/10;
        
        % Calculate distance from left two to right one
        L2R1 = sqrt((R1x - L2x)^2 + (R1y - L2y)^2);
        a = L2R1/10;
        
        % Calculate right step width
        semi_perimeter = (a+b+c)/2;
        right_step_width(i,1) = (2*sqrt(semi_perimeter*(semi_perimeter - a)*(semi_perimeter - b)*(semi_perimeter - c)))/b;
        
        % Calculate right step length
        right_step_length(i,1) = real(sqrt((b^2) - (right_step_width(i,1)^2)));
    end
    
    % Set limits for the foot placement plot for consistent scaling.
    ylim([-100000 100000]);
    xlim([-100000 100000]);
    
    %% Temporal/Temporophasic Parameters
    
    % Cadence - The number of steps per minute, also referred to as step rate.
    % Calculated as the number of steps divided by time walking.
    
    
    step_locs = vertcat(right_contact_locs,left_contact_locs);
    cadence = mean(height(step_locs)/4); % Divided by 4 minutes walking
    
    % Step time (seconds) - The time elapsed from the initial contact of one
    % foot to the initial contact of the opposite foot. Calculated as time
    % difference between heel strikes of each foot.
    step_locs = sort(step_locs(:,1),'ascend');
    step_time = diff(step_locs/sampling_rate);
    
    % Stride times - Left foot
    left_stride_time = left_contact_locs/sampling_rate;
    left_stride_time = diff(left_stride_time);
    
    % Stride times - Right foot
    right_stride_time = right_contact_locs/sampling_rate;
    right_stride_time = diff(right_stride_time);
    
    % Left Stance Time (seconds) - The time elapsed between the first and last
    % contacts of a single footfall (stance phase is the part of each gait
    % cycle where weight is supported and begins at heel contact and ends at
    % toe off of the same foot)
    if length(left_contact_locs) == length(left_toeoff_locs) & left_contact_locs(end) > left_toeoff_locs(end)
        for j = 1:length(left_contact_locs)-1
            left_stance_time(j,:) = abs((left_toeoff_locs(j+1) - left_contact_locs(j))/sampling_rate);
        end
    elseif length(left_contact_locs) == length(left_toeoff_locs) & left_contact_locs(end) < left_toeoff_locs(end)
        for j = 1:length(left_toeoff_locs)-1
            left_stance_time(j,:) = abs((left_toeoff_locs(j) - left_contact_locs(j+1))/sampling_rate);
        end
    elseif length(left_contact_locs) < length(left_toeoff_locs) & left_contact_locs(end) < left_toeoff_locs(end)
        for j = 1:length(left_contact_locs)
            left_stance_time(j,:) = abs((left_toeoff_locs(j+1) - left_contact_locs(j))/sampling_rate);
        end
    elseif length(left_contact_locs) < length(left_toeoff_locs) & left_contact_locs(end) > left_toeoff_locs(end)
        for j = 1:length(left_contact_locs)
            left_stance_time(j,:) = abs((left_toeoff_locs(j+1) - left_contact_locs(j))/sampling_rate);
        end
    elseif length(left_contact_locs) > length(left_toeoff_locs) & left_contact_locs(end) > left_toeoff_locs(end)
        for j = 1:length(left_contact_locs)-1
            left_stance_time(j,:) = abs((left_toeoff_locs(j) - left_contact_locs(j))/sampling_rate);
        end
    elseif length(left_contact_locs) > length(left_toeoff_locs) & left_contact_locs(end) < left_toeoff_locs(end)
        for j = 1:length(left_toeoff_locs)
            left_stance_time(j,:) = abs((left_toeoff_locs(j) - left_contact_locs(j+1))/sampling_rate);
        end
    end
    
    % Right Stance Time (seconds)
    if length(right_contact_locs) == length(right_toeoff_locs) & right_contact_locs(end) > right_toeoff_locs(end)
        for j = 1:length(right_contact_locs)-1
            right_stance_time(j,:) = abs((right_toeoff_locs(j+1) - right_contact_locs(j))/sampling_rate);
        end
    elseif length(right_contact_locs) == length(right_toeoff_locs) & right_contact_locs(end) < right_toeoff_locs(end)
        for j = 1:length(right_toeoff_locs)-1
            right_stance_time(j,:) = abs((right_toeoff_locs(j) - right_contact_locs(j+1))/sampling_rate);
        end
    elseif length(right_contact_locs) < length(right_toeoff_locs) & right_contact_locs(end) < right_toeoff_locs(end)
        for j = 1:length(right_contact_locs)
            right_stance_time(j,:) = abs((right_toeoff_locs(j+1) - right_contact_locs(j))/sampling_rate);
        end
    elseif length(right_contact_locs) < length(right_toeoff_locs) & right_contact_locs(end) > right_toeoff_locs(end)
        for j = 1:length(right_contact_locs)
            right_stance_time(j,:) = abs((right_toeoff_locs(j+1) - right_contact_locs(j))/sampling_rate);
        end
    elseif length(right_contact_locs) > length(right_toeoff_locs) & right_contact_locs(end) > right_toeoff_locs(end)
        for j = 1:length(right_contact_locs)-1
            right_stance_time(j,:) = abs((right_toeoff_locs(j) - right_contact_locs(j))/sampling_rate);
        end
    elseif length(right_contact_locs) > length(right_toeoff_locs) & right_contact_locs(end) < right_toeoff_locs(end)
        for j = 1:length(right_toeoff_locs)
            right_stance_time(j,:) = abs((right_toeoff_locs(j) - right_contact_locs(j+1))/sampling_rate);
        end
    end
    
    % Left Swing Time (seconds) - The time elapsed between the last contact of the
    % current footfall and the first touch of the following footfall of the
    % same foot(swing phase starts with toe off and ends with first contact of
    % the same foot). Here we are choosing to calculate as stride time
    % minus stance time
    for m = 1:length(left_stride_time)
        left_swing_time(m,:) = left_stride_time(m) - left_stance_time(m);
    end
    
    % Right Swing Time (seconds)
    for m = 1:length(right_stride_time)
        right_swing_time(m,:) = right_stride_time(m) - right_stance_time(m);
    end
    
    % Creating vectors of the right and left heel contacts, then adding them
    % together. Rows with 1000 indicate single support and rows with 2000
    % indicate double support.
    right_contact = dat(:,245);
    left_contact = dat(:,154);
    total_contacts = left_contact + right_contact;
    
    for n = 1:(length(right_contact_locs)-1)
        
        % Find first heelstrike
        start = right_contact_locs(n,1); % first heelstrike
        
        % Find second heelstrike which is the end of one stride cycle
        stop = right_contact_locs(n+1,1);
        
        % Creating many gait cycles that contain the number of rows between each
        % consecutive start and stop
        cycle(1:(stop-start+1),1) = total_contacts(start:stop,1);
        
        % Stride time - The time elapsed between the initial contacts of two
        % consecutive footfalls of the same foot.
        % Stride interval is the stop time minus the start time within each cycle
        stride_interval(n,1) = (stop - start)/200;
        
        % Double support time (seconds) - The total time both feet are
        % simultaneously in contact with the ground throughout the gait cycle
        % Double support is the number of rows, within each cycle, that equals 2000
        % (2000 is both contacts added together)
        double_support_time(n,1) = height((find(cycle == 2000)))/200;
        
        % Single support time (seconds) - The time elapsed between the last contact
        % of the opposing footfall and the first contact of the subsequent footfall
        % of the same foot (single support is a part of each gait cycle when only
        % one foot is in contact with the ground).
        % Single support is the number of rows that are not double support
        single_support_time(n,1) = stride_interval(n,1) - double_support_time(n,1);
        
        % Double support time (%Gait Cycle) - Double support time normalized to
        % stride time
        % Percent of gait cycle in double support
        pct_double(n,1) = (double_support_time(n,1)/stride_interval(n,1))*100;
        
        % Single support time (%Gait Cycle) - Single support time normalized to
        % stride time
        % Percent of gait cycle in single support
        pct_single(n,1) = (single_support_time(n,1)/stride_interval(n,1))*100;
        
    end
    
    % Percent of cycle spent in stance and swing left leg
    for q = 1:(length(left_swing_time))
        
        % Stance time (%Gait Cycle) - Stance time normalized to stride time
        left_pct_stance(q,1) = (left_stance_time(q)/left_stride_time(q))*100;
        
        % Swing time (%Gait Cycle) - Swing time normalized to stride time
        left_pct_swing(q,1) = (left_swing_time(q)/left_stride_time(q))*100;
        
    end
    
    % Percent of cycle spent in stance and swing right leg
    for p = 1:(length(right_swing_time))
        
        % Stance time (%Gait Cycle) - Stance time normalized to stride time
        right_pct_stance(p,1) = (right_stance_time(p)/right_stride_time(p))*100;
        
        % Swing time (%Gait Cycle) - Swing time normalized to stride time
        right_pct_swing(p,1) = (right_swing_time(p)/right_stride_time(p))*100;
        
    end
    
    %% Calculate Distance Traveled, Average Speed, Stride Speed
    
    % Calculate distance traveled in meters. Data is collected in mm then
    % scaled to meters. Using pelvis trajectory X and Y
    distance_traveled = sum(sqrt(diff(dat(:,316)).^2 + diff(dat(:,317)).^2))/1000;
    
    % Calculate average speed in m/s. Do not need to divide by the sampling
    % rate because the data has been cleaned to 240 seconds. Here, we are using
    % the value of the last row (240 seconds).
    average_speed = distance_traveled/dat(end,1);
    
    % Stride Speed. Speed of the limb from toe off to heel strike.
    left_stride_speed = (left_stride_length./left_stride_time)/100;
    right_stride_speed = (right_stride_length./right_stride_time)/100;
    
    %% Create a figure
    
    set(0, 'CurrentFigure', big_figure); % Make 'big_figure' the current figure.
    sgtitle(['Spatiotemporal Parameters: ', png_filename], 'interpreter', 'none') % Prevent lowercase titles
    
    subplot(4,3,1)
    plot(left_stride_time)
    xlim([0 300]);
    ylim([0.8 1.5]);
    title("Left Stride Time (s)")
    xlabel("Stride Number")
    ylabel("Time (s)")
    
    % subplot(4,3,[2;5])
    % Plot created starting on line 131
    
    subplot(4,3,3)
    plot(right_stride_time)
    xlim([0 300]);
    ylim([0.8 1.5]);
    title("Right Stride Time (s)")
    xlabel("Stride Number")
    ylabel("Time (s)")
    
    subplot(4,3,4)
    plot(left_stride_length)
    xlim([0 300]);
    ylim([80 280]);
    title("Left Stride Length (cm)")
    xlabel("Stride Number")
    ylabel("Length (cm)")
    
    subplot(4,3,6)
    plot(right_stride_length)
    xlim([0 300]);
    ylim([80 280]);
    title("Right Stride Length (cm)")
    xlabel("Stride Number")
    ylabel("Length (cm)")
    
    subplot(4,3,7)
    plot(left_step_length)
    xlim([0 300]);
    ylim([40 140]);
    title("Left Step Length (cm)")
    xlabel("Stride Number")
    ylabel("Length (cm)")
    
    subplot(4,3,[8;11])
    plot(dat(:,316), dat(:,317));
    ylim([-100000 100000]);
    xlim([-100000 100000]);
    title('Pelvis Trajectory');
    
    subplot(4,3,9)
    plot(right_step_length)
    xlim([0 300]);
    ylim([40 140]);
    title("Right Step Length (cm)")
    xlabel("Stride Number")
    ylabel("Length (cm)")
    
    subplot(4,3,10)
    plot(left_step_width)
    xlim([0 300]);
    ylim([0 70]);
    title("Left Step Width (cm)")
    xlabel("Stride Number")
    ylabel("Width (cm)")
    
    subplot(4,3,12)
    plot(right_step_width)
    xlim([0 300]);
    ylim([0 70]);
    title("Right Step Width (cm)")
    xlabel("Stride Number")
    ylabel("Width (cm)")
    
    set(gcf, 'Position', [400, 50, 1100, 925]); % Setting Figure Size;
    
    %% Save the Figure
    
    if save_option
        saveas(gcf,fullfile('Your filepath here',png));
    end
    
    close all % Close figures
    
    %% Export Table
    
    % Concatenate our data but add NaN to unused cells due to step_time having
    % twice the amout of rwos as stride_time
    spatiotemporal_variables = padcat(cadence, step_time, left_step_length,...
        right_step_length, left_step_width, right_step_width, left_stride_length,...
        right_stride_length, left_stride_time, right_stride_time, left_stance_time,...
        right_stance_time, left_swing_time, right_swing_time,single_support_time,...
        double_support_time, left_pct_stance, right_pct_stance, left_pct_swing,...
        right_pct_swing, pct_single, pct_double, average_speed, left_stride_speed,...
        right_stride_speed, distance_traveled);
    
    % Change to table and name variables for later use
    spatiotemporal_variables = array2table(spatiotemporal_variables);
    spatiotemporal_variables.Properties.VariableNames = {'cadence (steps/min)',...
        'step time (s)', 'left step length (cm)', 'right step length (cm)',...
        'left step width (cm)', 'right step width (cm)', 'left stride length (cm)',...
        'right stride length (cm)', 'left stride time (s)','right stride time (s)',...
        'left stance time (s)', 'right stance time (s)','left swing time (s)',...
        'right swing time (s)', 'single support_time (s)','double support time (s)',...
        'left pct stance (%GC)', 'right pct stance (%GC)','left pct swing (%GC)',...
        'right pct swing (%GC)', 'pct single (%GC)','pct double (%GC)',...
        'average speed (m/s)', 'left stride speed (m/s)', 'right stride speed (m/s)',...
        'distance traveled (m)'};
end

%end
