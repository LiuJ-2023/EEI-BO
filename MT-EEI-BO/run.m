addpath('Benchmark')
addpath('Tasks')
addpath('InitData')
for i = 1:9
    for j = 1:10
        [value_MT(i,j).value,plot_min_MT(i,j).value]=MT_EEI_BO(i);
    end
end