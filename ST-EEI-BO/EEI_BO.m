% Code of EEI-BO
% Implemented by Liu Jiao
% Ref. J. Liu, Y. Wang, G. Sun and T. Pang, "Solving Highly Expensive Optimization Problems via Evolutionary Expected Improvement," 
% in IEEE Transactions on Systems, Man, and Cybernetics: Systems, doi: 10.1109/TSMC.2023.3257030.
% Email: jiao.liu@ntu.edu.sg; liuj_csu@126.com
function [plot_cruve,DB] = EEI_BO(func_num)
Lambda = 500;
M = 50;
eta = 0.1;
%% Initialization
% Initialize population and database
load POP_10 % We provode the same initial database for fair comparison about the convergence
[~,Dim] = size(POP);
DB.x = POP;
DB.y = Evaluate(POP,func_num);
[DB.y_best,idx_min] = min(DB.y);
DB.x_best = DB.x(idx_min,:);

% Initialize CMA-ES
Mu = mean(POP);
Sigma = 50;
pc = 0*ones(d,1); 
ps = 0*ones(d,1);
B = eye(d); 
D = eye(d); 
C = B*D*transpose(B*D);
t = 0;
tmax = 30; 

Up = up;
Dn = dn;
plot_cruve = [DB.y_best];
while t < tmax
%% Build GP models
    Model_GP = fitrgp(DB.x,DB.y,'KernelFunction','squaredexponential','Sigma',0.1);
    ghxd=real(sqrt(DB.x.^2*ones(size(DB.x'))+ones(size(DB.x))*(DB.x').^2-2*DB.x*(DB.x')));
    spr=max(max(ghxd))/(d*length(DB.y))^(1/Dim);
    Model_RBF = newrbe(DB.x',DB.y',spr);    

%% Evolution of CMA-ES
    V = Mu + Sigma*mvnrnd(zeros(1,Dim),C,Lambda);
    V = (V >= Up).*Up + (V < Up).*V;
    V = (V <= Dn).*Dn + (V > Dn).*V;
    Predict_V = sim(Model_RBF,V');
    [Mu,Sigma,pc,ps,B,C,D] = CMAES(V,Predict_V,Mu,Sigma,pc,ps,B,C,D,M);
    Mu = (Mu >= Up).*Up + (Mu < Up).*Mu;
    Mu = (Mu <= Dn).*Dn + (Mu > Dn).*Mu;

%% Optimize EEI
    Sigma_Real = Sigma^2*C;
    x_new = DE(Model_GP,Mu,Sigma_Real,Up,Dn,DB.y_best);
    x_new = (x_new >= Up).*Up + (x_new < Up).*x_new;
    x_new = (x_new <= Dn).*Dn + (x_new > Dn).*x_new;
    y_new = Evaluate(x_new,func_num);

%% Update Database
    DB.x = [DB.x;x_new];
    DB.y = [DB.y;y_new];
    [DB.y_best,idx_min] = min(DB.y);
    DB.x_best = DB.x(idx_min,:);
    t = t+1;
    
%% Print
    fprintf([num2str((DB.y_best)),'\n'])
    plot_cruve = [plot_cruve,DB.y_best];
end

% Differential evaluation for optimizing acquisition function
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
        % Crossover
        u(i,:) = v(i,:).*(rj<0.9) + x(i,:).*(rj>=0.9);
        % Repair
        u(i,:) = (u(i,:) >= up).*up + (u(i,:) < up).*u(i,:);
        u(i,:) = (u(i,:) <= dn).*dn + (u(i,:) > dn).*u(i,:);    
        % Evaluation
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

% Evolution of CMA-ES
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

% Construction of EEI
function y = obj(x,f_min,dmodel,mu,C)
[~,d] = size(x);
EI = Infill_Standard_GP_EI(x, dmodel, f_min);
P = (1/(det(C)*(2*pi)^(d/2)))*exp(-0.5*(x - mu)*(C^-1)*(x - mu)');
y = EI*P;