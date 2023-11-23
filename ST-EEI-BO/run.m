for j = 1:6
    for i = 1:5
        [y_plot_EEI(i,:),y_best_EEI{i}] = EEI_BO(j);
        [y_plot(i,:),y_best{i}] = EI_BO(j);
    end
    plot_mean_(j,:) = mean(y_plot);
    plot_mean_EEI(j,:) = mean(y_plot_EEI);
end