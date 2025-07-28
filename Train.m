close all;
clearvars;
clc;
%% Train Parameters 测试参数
max_episode = 1000;     % Maximum number of episodes 最大回合数1000
max_step = 2000;        % Maximum number of steps in one episode 一个回合的最大步数2000
frequency = 1;          % Update frequency 更新频率
buffer_size = 1000000;  % Maximum size of buffer 缓冲区的最大尺寸
batch_size = 256;       % Batch size 批次尺寸
%% Build Environment 创建地图环境
env = Environment([0; 20], ...      % X-axis limit of the map 地图的x轴极限
                  [0; 20], ...      % Y-axis limit of the map 地图的x轴极限
                  [19; 19; 1], ...  % Goal (position and radius) 目标终点(中心和半径)
                  1, ...            % Is there any obstacle (1 Yes, 0 No) 是否有障碍物(1是，0否)
                  [6, 12, 2, 4; ...
                   6, 4, 2, 4; ...
                   12, 12, 2, 4; ...
                   12, 4, 2, 4; ...
                   ], ...           % Obstacle 障碍物位置和长宽
                  60, ...           % Number of lidar beam 激光雷达光束数量
                  8);               % Detection range of lidar 激光雷达的检测范围
%% Build Agent 创建代理
agent = Agent(3 + 2 + env.Num_rays, ...  % State 状态
              2, ...                     % Action 动作
              32, ...                    % Layer size 层尺寸
              0.001, ...                 % Learning speed of actor networks 演员网络的学习速度
              0.001, ...                 % Learning speed of the critic network 评论家网络的学习速度
              0.0003, ...                % Learning speed of entropy 熵的学习速度
              0.99, ...                  % Discount factor 折扣系数
              0.005); ...                % Target update coefficient 目标更新系数
%% Input Agent and Buffer 输入代理和缓冲区
% load('Trained.mat');
%% Create the remaining parts 创建其余部分
buffer = zeros(agent.StateSize*2 + agent.ActionSize + 2, buffer_size);  % Buffer are 0 缓冲区全0
buffer_count = 1;                                                       % Calculate buffer 计算缓冲区
rewardSave = zeros(1, max_episode);                                     % Save rewards 保存奖励积分
%% Train
for i = 1:max_episode
    %% Reset state 重置状态
    position = [1; 1; pi/4];
    lidarData = env.readLidar(position);
    p_g = (env.Goal(1:2) - position(1:2))./[env.Limx(2); env.Limy(2)];
    state = [position./[env.Limx(2); env.Limy(2); pi]; p_g; lidarData/env.Max_distance];
    gamma = 1;
    score = 0;
    entropy = 0;
    for j = 1:max_step
        %% Select and execute actions 选择并执行动作
        [action, logProb] = agent.selectAction(state);
        [nextState, reward, isDone] = env.step(state, action);
        buffer(:, max(1, mod(buffer_count, buffer_size))) = [state; action; reward; nextState; isDone];
        %% Update weights 更新权重
        if buffer_count > batch_size && mod(j, frequency) == 0
            % Obtain batch 获得批次
            randomIndices = randperm(min(buffer_size, buffer_count), batch_size);
            batch = buffer(:, randomIndices);
            % Update 更新
            agent = agent.updateCriticQ1(batch);
            agent = agent.updateCriticQ2(batch);
            agent = agent.updateActor(batch);
            agent = agent.updateTemperature(batch);
            agent = agent.updateTargetQ1();
            agent = agent.updateTargetQ2();
        end
        %% Check Critic and Target 检查评论网络和目标权重
        if j == 1
            [Q1, ~] = agent.criticForward(agent.CriticQ1Weights, state, action);
            [Q2, ~] = agent.criticForward(agent.CriticQ2Weights, state, action);
            [T1, ~] = agent.criticForward(agent.TargetQ1Weights, state, action);
            [T2, ~] = agent.criticForward(agent.TargetQ2Weights, state, action);
        end
        %% Draw 绘图
        % env.plot(state);
        %% Enter a new state 进入新状态
        score = score + gamma*reward;
        entropy = entropy - gamma*logProb;
        gamma = gamma*agent.Gamma;
        buffer_count = buffer_count + 1;
        state = nextState;
        %% Check stop conditions 检查停止条件
        if isDone == 1
            break;
        end
    end
    %% Save mat 保存数据
    rewardSave(i) = score;
    save('trained_data.mat', 'env', 'agent', 'buffer', 'buffer_count', 'rewardSave');
    %% Output result 显示结果
    fprintf('Episode: %-6d Step: %-5d Total Step: %-7d Score: %-8.2f Entropy: %-8.2f Alpha: %-8.4f Q1: %-8.2f Q2: %-8.2f T1: %-8.2f T2: %-8.2f\n', ...
             i, j, buffer_count - 1, score, entropy, agent.Alpha, Q1, Q2, T1, T2);
    %回合数 步数 总步数 得分 熵 代理 Q1 Q2 T1 T2
end
