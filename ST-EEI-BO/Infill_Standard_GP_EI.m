function obj = Infill_Standard_GP_EI(x, Kriging_model, f_min)
[y,mse] = predict(Kriging_model,x);
% s=sqrt(max(0,mse));
s = max(0,mse);
% calcuate the EI value
EI=(f_min-y).*Gaussian_CDF((f_min-y)./s)+s.*Gaussian_PDF((f_min-y)./s);
% this EI needs to be maximized
obj=-EI;

end