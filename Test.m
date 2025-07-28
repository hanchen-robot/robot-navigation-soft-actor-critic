close all;
clearvars;
clc;
%% Test Parameters 测试参数
max_episode = 1000;     % Maximum number of episodes 最大回合数
max_step = 2000;        % Maximum number of steps in one episode 一个回合的最大步数
frequency = 1;          % Update frequency 更新频率
buffer_size = 100000;   % Maximum size of buffer 缓冲区的最大尺寸
batch_size = 256;       % Batch size 批次尺寸
%% Input Agent and Buffer 输入代理和缓冲区
load('trained_data.mat');
speed = zeros(1, max_episode);
% %% Create the VideoWriter object with the highest resolution 创建分辨率最高的VideoWriter对象
% v = VideoWriter('Video 19 19.avi');
% v.Quality = 100; % Highest resolution
% v.FrameRate = 30; % Frames per second
% %% Create a new graphic with a larger size 创建更大尺寸的新图形
% figure('Position', [0, 0, 1080, 1080]); % Figure size
% %% Open VideoWriter 打开VideoWriter录制
% open(v);
%% Test
for i = 1:max_episode
    %% Reset state
    path = 0;
    position = [1; 1; pi/4];
    lidarData = env.readLidar(position);
    U_rep = env.repulsion(lidarData);
    p_g = (env.Goal(1:2) - position(1:2))./[env.Limx(2); env.Limy(2)];
    state = [position./[env.Limx(2); env.Limy(2); pi]; p_g; U_rep];
    x = position(1);
    y = position(2);
    gamma = 1;
    score = 0;
    entropy = 0;
    for j = 1:max_step
        %% Select and execute actions 选择并执行动作
        [action, logProb] = agent.selectAction(state);
        [nextState, reward, isDone] = env.step(state, action);
        %% Test Critic and Target 测试评论家网络和目标
        if j == 1
            [Q1, ~] = agent.criticForward(agent.CriticQ1Weights, state, action);
            [Q2, ~] = agent.criticForward(agent.CriticQ2Weights, state, action);
            [T1, ~] = agent.criticForward(agent.TargetQ1Weights, state, action);
            [T2, ~] = agent.criticForward(agent.TargetQ2Weights, state, action);
        end
        %% Draw 画图
        path = path + norm(nextState(1:2).*[env.Limx(2); env.Limy(2)] - state(1:2).*[env.Limx(2); env.Limy(2)]);
        x = cat(2, x, nextState(1)*env.Limx(2));
        y = cat(2, y, nextState(2)*env.Limy(2));
        % env.plot(state, x, y);
        % drawnow;
        % frame = getframe(gcf);
        % pause(0.01);
        % writeVideo(v, frame);
        %% Enter a new state 准备进入新状态
        score = score + gamma*reward;
        entropy = entropy - gamma*logProb;
        gamma = gamma*agent.Gamma;
        buffer_count = buffer_count + 1;
        state = nextState;
        %% Stop condition 停止条件
        if isDone == 1
            break;
        end
    end
    %% Computation speed 计算速度
    speed(:, i) = path/(j*0.1);
    %% Output result 显示结果
    fprintf('Episode: %-6d Step: %-5d Score: %-8.2f Entropy: %-8.2f Alpha: %-8.4f Q1: %-8.2f Q2: %-8.2f T1: %-8.2f T2: %-8.2f Speed: %.2f (m/s)\n', ...
             i, j, score, entropy, agent.Alpha, Q1, Q2, T1, T2, speed(i));
    %回合数 步数 得分 熵 代理 Q1 Q2 T1 T2 速度
end
% %% Close VideoWriter 关闭VideoWriter
% close(v);
%% Mean speed 平均速度
speedMean = mean(speed);
speedMax = max(speed);
speedMin = min(speed);
%% Draw map 绘制地图
figure(1);
env.plotMap;
%% Draw path 绘制机器人路径
figure(2);
env.plotPath(x, y);
%% Draw an average reward graph 绘制平均奖励积分图
figure(3);
rewardAverage = zeros(size(rewardSave));
for e = 1:size(rewardSave, 2)
    rewardAverage(e) = mean(rewardSave(max(1, e - 200):e));
end
plot(rewardSave, 'b');
hold on;
plot(rewardAverage, 'r', 'LineWidth', 3);
grid on;
xlim([0, size(rewardSave, 2)]);
ylim([0, 1200]);
legend('Average Reward (AR)', 'Integral Calculation of AR');
xlabel('Sample');
ylabel('Reward');
title('Average Reward Result');
saveas(gcf, 'Reward.png'); % Save as PNG 保存为PNG
