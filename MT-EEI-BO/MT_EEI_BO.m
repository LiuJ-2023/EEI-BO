% Code of EEI-BO+
% Implemented by Liu Jiao
% Ref. J. Liu, Y. Wang, G. Sun and T. Pang, "Solving Highly Expensive Optimization Problems via Evolutionary Expected Improvement," 
% in IEEE Transactions on Systems, Man, and Cybernetics: Systems, doi: 10.1109/TSMC.2023.3257030.
% Email: jiao.liu@ntu.edu.sg; liuj_csu@126.com
function [plot_min,DB] = MT_EEI_BO(index)
if index == 1
    load Init1
elseif index == 2
    load Init2
elseif index == 3
    load Init3
elseif index == 4
    load Init4
elseif index == 5
    load Init5
elseif index == 6
    load Init6
elseif index == 7
    load Init7
elseif index == 8
    load Init8
elseif index == 9
    load Init9
end
modeflag = 1;
count = 1;
while t < tmax
%% Build surrogate models
    dmodel_GP{1} = fitrgp(DB(1).x,DB(1).y,'KernelFunction','squaredexponential','Sigma',0.001);
    dmodel_GP{2} = fitrgp(DB(2).x,DB(2).y,'KernelFunction','squaredexponential','Sigma',0.001);

%% Evolution of CMA-ES
    % Task1
    V1 = Mu1 + Sigma1*mvnrnd(zeros(1,d1),C1,200);
    V1 = (V1 >= Tasks(1).Ub).*Tasks(1).Ub + (V1 < Tasks(1).Ub).*V1;
    V1 = (V1 <= Tasks(1).Lb).*Tasks(1).Lb + (V1 > Tasks(1).Lb).*V1;
    Predict_V1 = Prediction(V1,dmodel_GP{1});%
    [Mu1,Sigma1,pc1,ps1,B1,C1,D1] = CMAES(V1,Predict_V1,Mu1,Sigma1,pc1,ps1,B1,C1,D1,M);
    Mu1 = (Mu1 >= Tasks(1).Ub).*Tasks(1).Ub + (Mu1 < Tasks(1).Ub).*Mu1;
    Mu1 = (Mu1 <= Tasks(1).Lb).*Tasks(1).Lb + (Mu1 > Tasks(1).Lb).*Mu1;

    % Task2
    V2 = Mu2 + Sigma2*mvnrnd(zeros(1,d2),C2,200);
    V2 = (V2 >= Tasks(2).Ub).*Tasks(2).Ub + (V2 < Tasks(2).Ub).*V2;
    V2 = (V2 <= Tasks(2).Lb).*Tasks(2).Lb + (V2 > Tasks(2).Lb).*V2;
    Predict_V2 = Prediction(V2,dmodel_GP{2});%
    [Mu2,Sigma2,pc2,ps2,B2,C2,D2] = CMAES(V2,Predict_V2,Mu2,Sigma2,pc2,ps2,B2,C2,D2,M);
    Mu2 = (Mu2 >= Tasks(2).Ub).*Tasks(2).Ub + (Mu2 < Tasks(2).Ub).*Mu2;
    Mu2 = (Mu2 <= Tasks(2).Lb).*Tasks(2).Lb + (Mu2 > Tasks(2).Lb).*Mu2;

%% Calculate population distribution for different tasks
    % The case that two tasks have the same dimension
    if d1 == d2  
        Mu2_Norm = (Mu2 - Tasks(2).Lb)./(Tasks(2).Ub - Tasks(2).Lb);
        Mu1_New  = Mu2_Norm.*(Tasks(1).Ub - Tasks(1).Lb) + Tasks(1).Lb;
        C2_Norm  = diag(1./(Tasks(2).Ub - Tasks(2).Lb))*Sigma2^2*C2*diag(1./(Tasks(2).Ub - Tasks(2).Lb))';
        C1_New   = diag(Tasks(1).Ub - Tasks(1).Lb)'*C2_Norm.*diag(Tasks(1).Ub - Tasks(1).Lb);

        Mu1_Norm = (Mu1 - Tasks(1).Lb)./(Tasks(1).Ub - Tasks(1).Lb);
        Mu2_New  = Mu1_Norm.*(Tasks(2).Ub - Tasks(2).Lb) + Tasks(2).Lb;
        C1_Norm  = diag(1./(Tasks(1).Ub - Tasks(1).Lb))*Sigma1^2*C1*diag(1./(Tasks(1).Ub - Tasks(1).Lb))';
        C2_New   = diag(Tasks(2).Ub - Tasks(2).Lb)'*C1_Norm.*diag(Tasks(2).Ub - Tasks(2).Lb);
        
    % The case that two tasks have the different dimension
    elseif d1 > d2 
        Mu2_Norm = (Mu2 - Tasks(2).Lb)./(Tasks(2).Ub - Tasks(2).Lb);
        Add_Vec  = rand([1,d1-d2]);
        Mu2_Norm = [Mu2_Norm,Add_Vec];
        Mu1_New  = Mu2_Norm.*(Tasks(1).Ub - Tasks(1).Lb) + Tasks(1).Lb;
        C2_Norm  = diag(1./(Tasks(2).Ub - Tasks(2).Lb))*Sigma2^2*C2*diag(1./(Tasks(2).Ub - Tasks(2).Lb))';
        C1_Base  = eye(d1);
        C1_Base(1:d2,1:d2) = C2_Norm;
        C1_Trans = diag(Tasks(1).Ub - Tasks(1).Lb)'*C1_Base.*diag(Tasks(1).Ub - Tasks(1).Lb);
        C1_New = C1_Trans;

        Mu1_Norm = (Mu1 - Tasks(1).Lb)./(Tasks(1).Ub - Tasks(1).Lb);
        Mu1_Norm = Mu1_Norm(:,1:d2);
        Mu2_New  = Mu1_Norm.*(Tasks(2).Ub - Tasks(2).Lb) + Tasks(2).Lb;
        C1_Norm  = diag(1./(Tasks(1).Ub - Tasks(1).Lb))*Sigma1^2*C1*diag(1./(Tasks(1).Ub - Tasks(1).Lb))';
        C1_Norm = C1_Norm(1:d2,1:d2);
        C2_New   = diag(Tasks(2).Ub - Tasks(2).Lb)'*C1_Norm.*diag(Tasks(2).Ub - Tasks(2).Lb);
    
    % The case that two tasks have the different dimension
    elseif d1 < d2 
        Mu2_Norm = (Mu2 - Tasks(2).Lb)./(Tasks(2).Ub - Tasks(2).Lb);
        Mu2_Norm = Mu2_Norm(:,1:d1);
        Mu1_New  = Mu2_Norm.*(Tasks(1).Ub - Tasks(1).Lb) + Tasks(1).Lb;
        C2_Norm  = diag(1./(Tasks(2).Ub - Tasks(2).Lb))*Sigma2^2*C2*diag(1./(Tasks(2).Ub - Tasks(2).Lb))';
        C2_Norm  = C2_Norm(1:d1,1:d1);
        C1_New   = diag(Tasks(1).Ub - Tasks(1).Lb)'*C2_Norm.*diag(Tasks(1).Ub - Tasks(1).Lb);
        
        Mu1_Norm = (Mu1 - Tasks(1).Lb)./(Tasks(1).Ub - Tasks(1).Lb);
        Add_Vec  = rand([1,d2-d1]);
        Mu1_Norm  = [Mu1_Norm,Add_Vec];
        Mu2_New  = Mu1_Norm.*(Tasks(2).Ub - Tasks(2).Lb) + Tasks(2).Lb;
        C1_Norm  = diag(1./(Tasks(1).Ub - Tasks(1).Lb))*Sigma1^2*C1*diag(1./(Tasks(1).Ub - Tasks(1).Lb))';
        C1_Base  = eye(d2);
        C1_Base(1:d1,1:d1) = C1_Norm;
        C2_Trans   = diag(Tasks(2).Ub - Tasks(2).Lb)'*C1_Base.*diag(Tasks(2).Ub - Tasks(2).Lb);
        C2_New = C2_Trans;
    end
    
%% Optimize EEI and generate the query solution
    % No transfer case
    if modeflag == 1
        Mu1_Opt = Mu1;
        C1_Opt = Sigma1^2*C1;
        x_new1 = DE(dmodel_GP{1},Mu1_Opt,C1_Opt,Tasks(1).Ub,Tasks(1).Lb,DB(1).ymin);
        x_new1 = (x_new1 >= Tasks(1).Ub).*Tasks(1).Ub + (x_new1 < Tasks(1).Ub).*x_new1;
        x_new1 = (x_new1 <= Tasks(1).Lb).*Tasks(1).Lb + (x_new1 > Tasks(1).Lb).*x_new1;
        y_new1 = feval(Tasks(1).fnc,x_new1);
        DB(1).x = [DB(1).x;x_new1];
        DB(1).y = [DB(1).y;y_new1];
        DB(1).ymin = min(DB(1).y);
        plot_min(1).y = [plot_min(1).y,DB(1).ymin];
        
        Mu2_Opt = Mu2;
        C2_Opt = Sigma2^2*C2;
        x_new2 = DE(dmodel_GP{2},Mu2_Opt,C2_Opt,Tasks(2).Ub,Tasks(2).Lb,DB(2).ymin);
        x_new2 = (x_new2 >= Tasks(2).Ub).*Tasks(2).Ub + (x_new2 < Tasks(2).Ub).*x_new2;
        x_new2 = (x_new2 <= Tasks(2).Lb).*Tasks(2).Lb + (x_new2 > Tasks(2).Lb).*x_new2;
        y_new2 = feval(Tasks(2).fnc,x_new2);
        DB(2).x = [DB(2).x;x_new2];
        DB(2).y = [DB(2).y;y_new2];
        DB(2).ymin = min(DB(2).y);
        plot_min(2).y = [plot_min(2).y,DB(2).ymin];
        
    % Transfer case
    elseif modeflag == 2       
        Mu1_Opt = Mu1_New;
        C1_Opt = C1_New;
        x_new1 = DE(dmodel_GP{1},Mu1_Opt,C1_Opt,Tasks(1).Ub,Tasks(1).Lb,DB(1).ymin);
        x_new1 = (x_new1 >= Tasks(1).Ub).*Tasks(1).Ub + (x_new1 < Tasks(1).Ub).*x_new1;
        x_new1 = (x_new1 <= Tasks(1).Lb).*Tasks(1).Lb + (x_new1 > Tasks(1).Lb).*x_new1;
        y_new1 = feval(Tasks(1).fnc,x_new1);
        DB(1).x = [DB(1).x;x_new1];
        DB(1).y = [DB(1).y;y_new1];
        DB(1).ymin = min(DB(1).y);
        plot_min(1).y = [plot_min(1).y,DB(1).ymin];

        Mu2_Opt = Mu2_New;
        C2_Opt = C2_New;
        x_new2 = DE(dmodel_GP{2},Mu2_Opt,C2_Opt,Tasks(2).Ub,Tasks(2).Lb,DB(2).ymin);
        x_new2 = (x_new2 >= Tasks(2).Ub).*Tasks(2).Ub + (x_new2 < Tasks(2).Ub).*x_new2;
        x_new2 = (x_new2 <= Tasks(2).Lb).*Tasks(2).Lb + (x_new2 > Tasks(2).Lb).*x_new2;
        y_new2 = feval(Tasks(2).fnc,x_new2);
        DB(2).x = [DB(2).x;x_new2];
        DB(2).y = [DB(2).y;y_new2];
        DB(2).ymin = min(DB(2).y);
        plot_min(2).y = [plot_min(2).y,DB(2).ymin];
    end
    
%% Update optimization mode
    if (modeflag == 1) && (count == 6)
        modeflag = 2;
        count = 0;
    elseif (modeflag == 2)
        modeflag = 1;
    end
    count = count + 1;
    t = t+2;
    fprintf([num2str(DB(1).ymin),'\n'])
    fprintf([num2str(DB(2).ymin),'\n'])
    fprintf([num2str(modeflag),'\n'])
end

% DE for the no transfer mode
function x_best = DE(dmodel,mu,C,up,dn,f_min)
[~,d] = size(dn);
x = (up - dn).*rand([30,d]) + dn;
for i = 1:30
    y(i,:) = obj(x(i,:),f_min,dmodel,mu,C);
end
for Iter = 1:200
    for i = 1:30
        rs = randperm(30,3);
        rj = rand(1,d);
        % DE/rand/1
        v(i,:) = x(rs(1),:) + 0.5*(x(rs(2),:) - x(rs(3),:));
        % Bin crossover
        u(i,:) = v(i,:).*(rj<0.9) + x(i,:).*(rj>=0.9);
        % Repair
        u(i,:) = (u(i,:) >= up).*up + (u(i,:) < up).*u(i,:);
        u(i,:) = (u(i,:) <= dn).*dn + (u(i,:) > dn).*u(i,:);
        % Evaluate
        y_off(i,:) = obj(u(i,:),f_min,dmodel,mu,C);
        % Selection
        if y_off(i,:) <= y(i,:)
            x(i,:) = u(i,:);
            y(i,:) = y_off(i,:);
        end
    end
end
[~,idx_best] = min(y);
x_best = x(idx_best,:);

% CMAES Update
function [x_elist_m,sigma,pc,ps,B,C,D] = CMAES(POP,y,mu,sigma,pc,ps,B,C,D,M)
[N,d] = size(POP);

% parameter setting: adaptation
cc = 4/(d+4); 
ccov = 2/(d+2^0.5)^2;
cs = 4/(d+4); 
damp = 1/cs + 1;
arweights = log((10+1)/2) - log(1:M); % for recombination
cw = sum(arweights)/norm(arweights);
chiN = d^0.5*(1-1/(4*d)+1/(21*d^2));

[~,idx_sort] = sort(y);
x_elist = POP(idx_sort(1:M),:);

x_elist_m = arweights*x_elist/sum(arweights);
dmu = x_elist_m - mu;

% Adapt covariance matrix
pc = (1-cc)*pc + (sqrt(cc*(2-cc))*cw) * dmu'/sigma; % Eq.(14)
C = (1-ccov)*C + ccov*pc*pc'; % Eq.(15)

% Adapt sigma
ps = (1-cs)*ps + (sqrt(cs*(2-cs))*cw) * (B*D^(-1)*B^(-1)*dmu')/sigma; % Eq.(16)
sigma = sigma * exp((norm(ps)-chiN)/chiN/damp); % Eq.(17)

% Update B and D from C
C=triu(C)+transpose(triu(C,1)); % enforce symmetry
[B,D] = eig(C);

% limit condition of C to 1e14 + 1
if max(diag(D)) > 1e14*min(diag(D))
    tmp = max(diag(D))/1e14 - min(diag(D));
    C = C + tmp*eye(N); 
    D = D + tmp*eye(N);
end
D = diag(sqrt(diag(D))); % D contains standard deviations now

% Adjust minimal step size
if sum(abs(dmu)) == 0
%     if rand<0.5
%         sigma = 1.2*sigma;
%     end
end

% Acquisition function
function y = obj(x,f_min,dmodel,mu,C)
[~,d] = size(x);
EI = Infill_Standard_GP_EI(x, dmodel, f_min);
P = (1/(det(C)*(2*pi)^(d/2)))*exp(-0.5*(x - mu)*(C^-1)*(x - mu)');
y = EI*P;

function y = Prediction(x,dmodel)
[y,rmse] = predict(dmodel,x);
