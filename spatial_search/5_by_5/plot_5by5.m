load("data.mat")

N = 5;
vis_times = [2, 4, 6];
vis_index = [5, 9, 13];


%-- Set default options --%
set(0,'DefaultAxesFontSize', 10, ...
      'DefaultAxesFontName', 'Helvetica',...
      'DefaultAxesLineWidth', 1.0);

%% Ideal subplots

for i = 1:3
    figure(i);
    idx = vis_index(i);
    time = vis_times(i);
    z_data = reshape(ideal_dist(idx,:), N, N);
    b = bar3(z_data);
    zlim([0 0.5])    % Set Z-axis range
    ax = gca;
    ax.FontSize = 10;
    %ax.XTickLabel = {'|00001\rangle','|00010\rangle','|00100\rangle','|01000\rangle', '|10000\rangle'};
    %ax.YTickLabel = {'|00001\rangle','|00010\rangle','|00100\rangle','|01000\rangle', '|10000\rangle'};
    ax.XTickLabelRotation = -30;
    title_text = sprintf('T = %d', time);
    title(title_text, 'Fontsize', 32)
    
    colormap(summer)
    clim([0 0.4])
    for k = 1:length(b)
        b(k).CData = b(k).ZData;
        b(k).FaceColor = 'interp';
    end
    colorbar();
    f = gcf;
    file_name = sprintf("search_5_ideal_%d.png", i);
    exportgraphics(f,file_name,'Resolution',300)
end


%% IonQ subplots

for i = 1:3
    figure(i);
    idx = vis_index(i);
    time = vis_times(i);
    z_data = reshape(ionq_freq(idx,:), N, N);
    b = bar3(z_data);
    zlim([0 0.3])    % Set Z-axis range
    ax = gca;
    ax.FontSize = 10;
    %ax.XTickLabel = {'|00001\rangle','|00010\rangle','|00100\rangle','|01000\rangle', '|10000\rangle'};
    %ax.YTickLabel = {'|00001\rangle','|00010\rangle','|00100\rangle','|01000\rangle', '|10000\rangle'};
    ax.XTickLabelRotation = -30;
    title_text = sprintf('T = %d', time);
    title(title_text, 'Fontsize', 32)
    
    colormap(parula)
    clim([0 0.15])
    for k = 1:length(b)
        b(k).CData = b(k).ZData;
        b(k).FaceColor = 'interp';
    end
    colorbar();
    f = gcf;
    file_name = sprintf("search_5_ionq_%d.png", i);
    exportgraphics(f,file_name,'Resolution',300)
end