classdef Environment
    %% Property Definition 定义属性
    properties
        Limx          % X-axis limit x轴范围
        Limy          % Y-axis limit y轴范围
        Goal          % Center and radius of the goal 目标终点的中心和半径
        IsObs         % Is there an obstacle (true, false) 确认是否有障碍物
        Obstacles     % Rectangular obstacle 矩形障碍物设置
        Num_rays      % Number of Lidar beams 激光雷达光束数量
        Max_distance  % Range of Lidar 激光雷达最大检测距离
        RobotRadius   % Radius of Robot 机器人数量
        d             % Distance between two wheels 机器人两轮之间距离
        dt            % Sample extraction time 采样提取时间
    end
    %% Method Definition 定义方法
    methods
        %% Initialization 初始化
        function obj = Environment(limx, limy, goal, isObs, obstacles, num_rays, max_distance)
            if nargin > 0
                obj.Limx = limx;
                obj.Limy = limy;
                obj.Goal = goal;
                obj.IsObs = isObs;
                obj.Obstacles = obstacles;
                obj.Num_rays = num_rays;
                obj.Max_distance = max_distance;
                obj.RobotRadius = 0.25;
                obj.d = 0.37;
                obj.dt = 0.1;
            end
        end
        %% Environmental State Transition Function 环境状态转移函数
        function [nextState, reward, isDone] = step(obj, state, action)
            % Get status and action 获取状态和动作
            x = state(1)*obj.Limx(2);
            y = state(2)*obj.Limy(2);
            theta = state(3)*pi;
            u1 = action(1);
            u2 = action(2);
            % Robot dynamics 机器人动力学方程
            x1 = x + obj.dt*(0.5*(u1 + u2)*cos(theta));
            y1 = y + obj.dt*(0.5*(u1 + u2)*sin(theta));
            theta1 = theta + obj.dt*((u1 - u2)*obj.d);
            % Check if it has been collided 检查它是否被碰撞
            lidarData1 = obj.readLidar([x1; y1; theta1]);
            isCollision = false;
            if min(lidarData1) < obj.RobotRadius
                isCollision = true;
            end
            % Distance to goal 到达终点距离
            oldDistanceToGoal = norm([x - obj.Goal(1); y - obj.Goal(2)]);
            newDistanceToGoal = norm([x1 - obj.Goal(1); y1 - obj.Goal(2)]);
            % Definition of cosAlpha 对cosAlpha定义
            a = [cos(theta1); sin(theta1)];
            b = obj.Goal(1:2) - [x1; y1];
            cosAlpha = a'*b/norm(b);
            % Check conditions and calculate reward scores 检查碰撞条件并计算奖励得分
            if isCollision
                reward = -1000;
                theta1 = theta1 + pi/2;
                isDone = 0;
            elseif newDistanceToGoal < obj.Goal(3)
                reward = 5000;
                isDone = 1;
            else
                reward = 50*(oldDistanceToGoal - newDistanceToGoal) + 5*cosAlpha;
                isDone = 0;
            end
            % Update state 更新状态
            position1 = [x1; y1; theta1];
            p_g1 = (obj.Goal(1:2) - position1(1:2))./[obj.Limx(2); obj.Limy(2)];
            nextState = [position1./[obj.Limx(2); obj.Limy(2); pi]; p_g1; lidarData1/obj.Max_distance];
        end
        %% U_rep Calculation 计算U_rep
        function U_rep = repulsion(obj, lidarData)
            U_rep = (1./lidarData - 1/obj.Max_distance).^2;
        end
        %% Read Lidar 获取激光雷达数据
        function lidarData = readLidar(obj, state)
            % Calculation of lidar value 计算激光雷达距离和角度
            lidarData = ones(obj.Num_rays, 1) * obj.Max_distance;
            angles = linspace(-pi/2, pi/2, obj.Num_rays);
            % Repeatedly read each laser beam 重复读取每个激光雷达光束
            for i = 1:obj.Num_rays
                % Linear equation of laser beam 激光雷达光束的线性方程
                angle = angles(i) + state(3);
                dx = cos(angle);
                dy = sin(angle); 
                % Find the intersection with the obstacle 找到与障碍物的交叉点
                if obj.IsObs == 1
                    for j = 1:size(obj.Obstacles, 1)
                        xc = obj.Obstacles(j, 1);
                        yc = obj.Obstacles(j, 2);
                        r = obj.Obstacles(j, 3);
                        % Find the intersection 搜索交叉点
                        a = dx^2 + dy^2;
                        b = 2*(dx*(state(1) - xc) + dy*(state(2) - yc));
                        c = (state(1) - xc)^2 + (state(2) - yc)^2 - r^2;
                        delta = b^2 - 4*a*c;
                        % If the intersection equation is experienced 如果遇到交叉点
                        if delta >= 0
                            t1 = (-b + sqrt(delta))/(2*a);
                            t2 = (-b - sqrt(delta))/(2*a);
                            if t1 > 0 && t1 <= obj.Max_distance
                                lidarData(i) = min(lidarData(i), t1);
                            end
                            if t2 > 0 && t2 <= obj.Max_distance
                                lidarData(i) = min(lidarData(i), t2);
                            end
                        end
                    end
                end
                % Find the intersection in the environment 在环境中找到交叉点
                environment_bounds = [obj.Limx(1), obj.Limy(1), obj.Limx(2) - obj.Limx(1), obj.Limy(2) - obj.Limy(1)];
                x_min = environment_bounds(1);
                y_min = environment_bounds(2);
                x_max = x_min + environment_bounds(3);
                y_max = y_min + environment_bounds(4);
                % Matrix Initiation 矩阵初始化
                t_values = [];
                % Search the intersection 查找交叉点
                if dx ~= 0
                    t1 = (x_min - state(1)) / dx;
                    y1 = state(2) + t1 * dy;
                    if y1 >= y_min && y1 <= y_max && t1 > 0 && t1 <= obj.Max_distance
                        t_values = cat(2, t_values, t1);
                    end 
                    t2 = (x_max - state(1)) / dx;
                    y2 = state(2) + t2 * dy;
                    if y2 >= y_min && y2 <= y_max && t2 > 0 && t2 <= obj.Max_distance
                        t_values = cat(2, t_values, t2);
                    end
                end
                % Find the intersection 找到交叉点
                if dy ~= 0
                    t3 = (y_min - state(2)) / dy;
                    x3 = state(1) + t3 * dx;
                    if x3 >= x_min && x3 <= x_max && t3 > 0 && t3 <= obj.Max_distance
                        t_values = cat(2, t_values, t3);
                    end
                    t4 = (y_max - state(2)) / dy;
                    x4 = state(1) + t4 * dx;
                    if x4 >= x_min && x4 <= x_max && t4 > 0 && t4 <= obj.Max_distance
                        t_values = cat(2, t_values, t4);
                    end
                end
                % Check for any intersections 检查是否有交叉点
                if ~isempty(t_values)
                    lidarData(i) = min(lidarData(i), min(t_values));
                end
                if lidarData(i) == 0
                    lidarData(i) = obj.Max_distance;
                end
            end
        end
        %% Draw 画图
        function plot(obj, state, x, y)
            clf;
            hold on;
            % Navigation task 导航任务
            position = state(1:3).*[obj.Limx(2); obj.Limy(2); pi];
            lidarData = obj.readLidar(position);
            % Draw robots and paths 绘制机器人和路径
            plot(x, y, 'LineWidth', 2);
            plot(position(1), position(2), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
            % Draw environmental bounds 绘制环境边界
            environment_bounds = [obj.Limx(1), obj.Limy(1), obj.Limx(2) - obj.Limx(1), obj.Limy(2) - obj.Limy(1)];
            rectangle('Position', environment_bounds, 'EdgeColor', 'k', 'LineWidth', 2);
            % Draw obstacles 绘制障碍物
            if obj.IsObs == 1
                for o = 1:size(obj.Obstacles, 1)
                    %viscircles(obj.Obstacles(o, 1:2), obj.Obstacles(o, 3), 'Color', 'r', 'LineWidth', 2);
                    rectangle('Position', obj.Obstacles(o, 1:4), 'FaceColor', '[0.5, 0.5, 0.5]', 'LineWidth', 2);
                    %viscircles([20 30],10,'color','r'); 绘制圆心坐标为(20,30),半径为10,轮廓颜色为红色
                    %rectangle('Position', [2, 3, 5, 4], 'FaceColor', 'r'); 绘制一个位于(2,3)处,宽度为5,高度为4的矩形
                end
            end
            % Draw the starting point 绘制起点
            plot(1, 1, 'bx', 'LineWidth', 2);
            % Draw the goal 绘制终点
            viscircles(obj.Goal(1:2)', obj.Goal(3), 'Color', 'r', 'LineWidth', 2);
            % Draw the lidar bem 绘制激光束
            delete(findall(gcf, 'Type', 'line', 'Color', 'b')); % Remove old lidar beams 移除旧的激光雷达光束
            angles = linspace(-pi/2, pi/2, obj.Num_rays);
            for i = 1:obj.Num_rays
                angle = angles(i) + position(3);
                end_x = position(1) + lidarData(i) * cos(angle);
                end_y = position(2) + lidarData(i) * sin(angle);
                plot([position(1), end_x], [position(2), end_y], 'b');
            end
            % Installation 定义图片
            axis equal;
            xlim(obj.Limx);
            ylim(obj.Limy);
            xlabel('X');
            ylabel('Y');
            title('Mobile Robots Avoid Obstacles');
            saveas(gcf, 'Test Obstacle.png'); % Save as PNG 保存为PNG
        end
        %% Draw Path 绘制路径
        function plotPath(obj, x, y)
            % Robot trajectory 机器人轨迹
            plot(x, y, 'LineWidth', 2);
            hold on;
            % Draw environmental bounds 绘制环境边界
            environment_bounds = [obj.Limx(1), obj.Limy(1), obj.Limx(2) - obj.Limx(1), obj.Limy(2) - obj.Limy(1)];
            rectangle('Position', environment_bounds, 'EdgeColor', 'k', 'LineWidth', 2);
            % Draw obstacles 绘制障碍物
            if obj.IsObs == 1
                for o = 1:size(obj.Obstacles, 1)
                    %viscircles(obj.Obstacles(o, 1:2), obj.Obstacles(o, 3), 'Color', 'r', 'LineWidth', 2);
                    rectangle('Position', obj.Obstacles(o, 1:4), 'FaceColor', '[0.5, 0.5, 0.5]', 'LineWidth', 2);
                end
            end
            % Draw the starting point 绘制起点
            plot(1, 1, 'bx', 'LineWidth', 2);
            % Draw the goal 绘制终点
            viscircles(obj.Goal(1:2)', obj.Goal(3), 'Color', 'r', 'LineWidth', 2);
            % Installation 定义图片
            xlim(obj.Limx);
            ylim(obj.Limy);
            xlabel('X');
            ylabel('Y');
            title('Robot Path After SAC Training');
            saveas(gcf, 'Test Path.png'); % Save as PNG 保存为PNG
        end
        %% Draw Map 绘制地图
        function plotMap(obj)
            % Draw environmental bounds 绘制环境边界
            environment_bounds = [obj.Limx(1), obj.Limy(1), obj.Limx(2) - obj.Limx(1), obj.Limy(2) - obj.Limy(1)];
            rectangle('Position', environment_bounds, 'EdgeColor', 'k', 'LineWidth', 2);
            % Draw obstacles 绘制障碍物
            if obj.IsObs == 1
                for o = 1:size(obj.Obstacles, 1)
                    %viscircles(obj.Obstacles(o, 1:2), obj.Obstacles(o, 3), 'Color', 'r', 'LineWidth', 2);
                    rectangle('Position', obj.Obstacles(o, 1:4), 'FaceColor', '[0.5, 0.5, 0.5]', 'LineWidth', 2);
                end
            end
            hold on;
            % Draw the starting point 绘制起点
            plot(1, 1, 'bx', 'LineWidth', 2);
            % Draw the goal 绘制终点
            viscircles(obj.Goal(1:2)', obj.Goal(3), 'Color', 'r', 'LineWidth', 2);
            % Installation 定义图片
            xlim(obj.Limx);
            ylim(obj.Limy);
            xlabel('X');
            ylabel('Y');
            title('Test Map for Mobile Robots');
            saveas(gcf, 'Test Map.png'); % Save as PNG 保存为PNG
        end
    end
end