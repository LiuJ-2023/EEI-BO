Lambda = 500;
M = 50;
eta = 0.1;
%% 初始化
% 初始化种群和数据集
N = 20;
Tasks = benchmark(index);
POP(1).x = (Tasks(1).Ub - Tasks(1).Lb).*rand([N,Tasks(1).dims]) + Tasks(1).Lb;
POP(2).x = (Tasks(2).Ub - Tasks(2).Lb).*rand([N,Tasks(2).dims]) + Tasks(2).Lb;
for i = 1:N
    POP(1).y(i,:) = feval(Tasks(1).fnc,POP(1).x(i,:));
    POP(2).y(i,:) = feval(Tasks(2).fnc,POP(2).x(i,:));
end
DB(1).x = POP(1).x;
DB(1).y = POP(1).y;
DB(1).ymin = min(DB(1).y);
[~,d1] = size(POP(1).x);
DB(2).x = POP(2).x;
DB(2).y = POP(2).y;
DB(2).ymin = min(DB(2).y);
[~,d2] = size(POP(2).x);

% 初始化CMA-ES参数
Mu1 = mean(POP(1).x);
Sigma1 = 0.5*mean(Tasks(1).Ub - Tasks(1).Lb);
pc1 = 0*ones(d1,1); 
ps1 = 0*ones(d1,1);
B1 = eye(d1); 
D1 = eye(d1); 
C1 = B1*D1*transpose(B1*D1);

Mu2 = mean(POP(2).x);
Sigma2 = 0.5*mean(Tasks(2).Ub - Tasks(2).Lb);
pc2 = 0*ones(d2,1); 
ps2 = 0*ones(d2,1);
B2 = eye(d2); 
D2 = eye(d2); 
C2 = B2*D2*transpose(B2*D2);

t = 1;
tmax = 100; 
flag = 1;

% Up = up;
% Dn = dn;
plot_min(1).y = DB(1).ymin;
plot_min(2).y = DB(2).ymin;