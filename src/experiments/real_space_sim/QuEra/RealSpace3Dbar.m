load("data.mat")

n = 6;
N = n + 1;


%-- Set default options --%
set(0,'DefaultAxesFontSize', 10, ...
      'DefaultAxesFontName', 'Helvetica',...
      'DefaultAxesLineWidth', 1.0);

%% Ideal subplots

for i = 1:3
    figure(i);
   
    time = sim_end_times(i);
    z_data = reshape(sim_dists(i,:,:), N, N);
    b = bar3(z_data);
    zlim([0 0.25])    % Set Z-axis range
    ax = gca;
    ax.FontSize = 10;
    %ax.XTickLabel = {'|00001\rangle','|00010\rangle','|00100\rangle','|01000\rangle', '|10000\rangle'};
    %ax.YTickLabel = {'|00001\rangle','|00010\rangle','|00100\rangle','|01000\rangle', '|10000\rangle'};
    ax.XTickLabelRotation = -30;
    title_text = sprintf('T = %.1f', time * 1e6);
    title(title_text, 'Interpreter', 'latex', 'Fontsize', 16)
    
    colormap(summer)
    clim([0 0.25])
    for k = 1:length(b)
        b(k).CData = b(k).ZData;
        b(k).FaceColor = 'interp';
    end
    colorbar();
    f = gcf;
    file_name = sprintf("sim_%d.png", i);
    exportgraphics(f,file_name,'Resolution',300)
end


%% QuEra subplots

for i = 1:3
    figure(i);
    time = quera_end_times(i);
    z_data = reshape(quera_dists(i,:,:), N, N);
    b = bar3(z_data);
    zlim([0 0.2])    % Set Z-axis range
    ax = gca;
    ax.FontSize = 10;
    %ax.XTickLabel = {'|00001\rangle','|00010\rangle','|00100\rangle','|01000\rangle', '|10000\rangle'};
    %ax.YTickLabel = {'|00001\rangle','|00010\rangle','|00100\rangle','|01000\rangle', '|10000\rangle'};
    ax.XTickLabelRotation = -30;
    title_text = sprintf('T = %.1f', time * 1e6);
    title(title_text, 'Interpreter', 'latex', 'Fontsize', 16)
    
    colormap(parula)
    clim([0 0.2])
    for k = 1:length(b)
        b(k).CData = b(k).ZData;
        b(k).FaceColor = 'interp';
    end
    colorbar();
    f = gcf;
    file_name = sprintf("quera_%d.png", i);
    exportgraphics(f,file_name,'Resolution',300)
end