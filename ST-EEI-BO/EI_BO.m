% EI-based BO for comparision
function [plot_cruve,DB] = EI_BO(func_num)
%% Initialization
load POP_10
DB.x = POP;
DB.y = Evaluate(POP,func_num);
[DB.y_best,idx_min] = min(DB.y);
DB.x_best = DB.x(idx_min,:);
t = 0;
tmax = 30; 

Up = up;
Dn = dn;
plot_cruve = [DB.y_best];
while t < tmax
%% Build GP models
    Model_GP = fitrgp(DB.x,DB.y,'KernelFunction','squaredexponential','Sigma',0.01);

%% Optimize EI
    x_new = DE(Model_GP,Up,Dn,DB.y_best);
    x_new = (x_new >= Up).*up + (x_new < Up).*x_new;
    x_new = (x_new <= Dn).*dn + (x_new > Dn).*x_new;
    y_new = Evaluate(x_new,func_num);

%% Update database
    DB.x = [DB.x;x_new];
    DB.y = [DB.y;y_new];
    [DB.y_best,idx_min] = min(DB.y);
    DB.x_best = DB.x(idx_min,:);
    t = t+1;
    
%% Print
    fprintf([num2str((DB.y_best)),'\n'])
    plot_cruve = [plot_cruve,DB.y_best];
end

% DE for optimizing acquisition
function x_best = DE(dmodel,up,dn,f_min)
[~,d] = size(dn);
x = (up - dn).*rand([30,d]) + dn;
for i = 1:30
    y(i,:) = obj(x(i,:),f_min,dmodel);
end
for Iter = 1:200
    for i = 1:30
        rs = randperm(30,3);
        rj = rand(1,d);
        % DE/rand/1
        v(i,:) = x(rs(1),:) + 0.5*(x(rs(2),:) - x(rs(3),:));
        % Crossover
        u(i,:) = v(i,:).*(rj<0.9) + x(i,:).*(rj>=0.9);
        % Repair
        u(i,:) = (u(i,:) >= up).*up + (u(i,:) < up).*u(i,:);
        u(i,:) = (u(i,:) <= dn).*dn + (u(i,:) > dn).*u(i,:);
        % Evaluation
        y_off(i,:) = obj(u(i,:),f_min,dmodel);
        % Ñ¡Ôñ
        if y_off(i,:) <= y(i,:)
            x(i,:) = u(i,:);
            y(i,:) = y_off(i,:);
        end
    end
end
[~,idx_best] = min(y);
x_best = x(idx_best,:);

% Acquisition function (EI)
function y = obj(x,f_min,dmodel)
[~,d] = size(x);
EI = Infill_Standard_GP_EI(x, dmodel, f_min);
y = EI;