%% 
close all
clear all
clc

%%% Antes de rodar precisar carregar a rede treinada 
%% Carregamento da Planilha de dados para treino
M = load('Valid_extra_EF_meg_10.dat');      %carrega o arquivo de dados%
M = M';                                     %calcula a matriz transposta
entrada = M(1:6,:);                         %define os dados de entrada da rede
saida = M(7,:);                             %define os dados de saída da rede
title_0 = 'MEG_10_%_Delta_P_[PSI]';

%% Carregamento dos parametros para a normalizacao dos dados
% Usando os mesmos parametros usados no treinamento das ANN
ent_norm_param = load('max_min_inputs_delta_P_para_norm.dat');
minentrada = ent_norm_param(2,:)';
maxentrada = ent_norm_param(1,:)';
saida_norm_param = load('max_min_outputs_K1_delta_P_para_norm.dat');
minsaida = saida_norm_param(2);
maxsaida = saida_norm_param(1);


%% Normalizacao dos dados - k+1
[ent_valid_norm] = tramnmx(entrada,minentrada,maxentrada); %normaliza dados%
[saida_valid_norm] = tramnmx(saida,minsaida,maxsaida); %normaliza dados%

%% Carregamento da RN
% Verificar o endereco completo clicando no arq
load('FF ANN MQI2104 k1_tansig_7_purelin_1_trainlm.mat')

%% Simulação de validacao
valid_norm = sim(net,ent_valid_norm(:,:));                  %simula com os dados de entrada do arquivo de teste%
valid_desnorm = postmnmx(valid_norm,minsaida,maxsaida);     %desnormaliza dados de saida%

%% Parametros de erro
r2_curva_k1 = calc_r2(valid_desnorm,saida);
sse_curva_k1 = sse(net, saida, valid_desnorm);
sse_curva_nor_k1 = sse(net, saida_valid_norm, valid_norm);
mse_curva_k1 = immse(saida, valid_desnorm);
rmse_curva_k1 = sqrt(mse_curva_k1);


%% Graficos do valid - k+1

%Criação da reta de ajuste linear para o treinamento
[coef_curva, error1] = polyfit(valid_desnorm,saida,1);     %define os coeficientes da reta de ajuste 1
Y_curva = polyval(coef_curva, valid_desnorm, error1);      %define a variável y da reta de ajuste
X_curva = valid_desnorm;                                   %define a variável x da reta de ajuste

%Criação da reta modelo
x= 0 : 1;
y = x;

%Grafico de correlacao entre os dados da simulcao e os da base
figure(1)                                   
subplot(2,1,1);                            
plot(valid_desnorm(1,:),saida(1,:),'or');            %gráfico da saída calculada x saída real
hold on                                     %permite que mais de uma curva seja plotada no mesmo gráfico
plot(X_curva,Y_curva,'-k');                 %plota a reta de melhor ajuste dos pontos (reta calculada)
plot(x,y,'--b'); %plota a reta modelo (y = x)title1 = strcat('Microbial Concentration Observed vs Calculated - Train R^2 =', num2str(r2_curve1)); %artificio para plotar texto e valores no título do gráfico
title1 = strcat('Delta P (k+1) [PSI] Observed vs Calculated - R^2 =', num2str(r2_curva_k1)); %artificio para plotar texto e valores no título do gráfico
title(title1,'FontSize',14); %título do primeiro gráfico
xlabel('\bf Delta P Calculated','FontSize',12); %legenda do eixo x, com fonte de tamanho 9
ylabel('\bf Delta P Observed','FontSize',12); %legenda do eixo y, com fonte de tamanho 9
legend('Train','Test', 'Location','northwest'); %legenda do gráfico no canto superior direito%
hold off

%Grafico de Comapracao - sobreposicao dos dados da simulcao e os da base
subplot(2,1,2);                     
plot(saida(1,:),'-k'); %grafico da saída real
hold on
plot(valid_desnorm(1,:),'vm');                              %gráfico da saída calculada
xlabel('\bf Time [s]','FontSize',12); %legenda do eixo x, com fonte de tamanho 9
ylabel('\bf ANN Delta P','FontSize',12); %legenda do eixo y, com fonte de tamanho 9
legend('Samples','ANN', 'Location','northwest'); %legenda do gráfico no canto superior direito%
title('Delta P (k+1) [PSI] Samples vs ANN','FontSize',14); %título do gráfico, com fonte do tamanho 11
hold off

print('-dtiff', '-r600', strcat(title_0,'_k_1',".tiff"))
savefig (strcat(title_0,'_k_1'));          %salva o gráfico como arquivo editável do matlab


%% Dados para o k+5

saida_norm_param = load('max_min_outputs_K5_delta_P_para_norm.dat');
minsaida = saida_norm_param(2);
maxsaida = saida_norm_param(1);

saida = M(11,:);
[saida_valid_norm] = tramnmx(saida,minsaida,maxsaida); %normaliza dados%

% Carregamento da RN
load('FF ANN MQI2104 k5_tansig_7_purelin_1_trainbr.mat')


%% Simulação de validacao
valid_norm = sim(net,ent_valid_norm(:,:));                  %simula com os dados de entrada do arquivo de teste%
valid_desnorm = postmnmx(valid_norm,minsaida,maxsaida);     %desnormaliza dados de saida%

%% Parametros de erro
r2_curva_k5 = calc_r2(valid_desnorm,saida);
sse_curva_k5 = sse(net, saida, valid_desnorm);
sse_curva_nor_k5 = sse(net, saida_valid_norm, valid_norm);
mse_curva_k5 = immse(saida, valid_desnorm);
rmse_curva_k5 = sqrt(mse_curva_k5);


%% Graficos do valid - k+1

%Criação da reta de ajuste linear para o treinamento
[coef_curva, error1] = polyfit(valid_desnorm,saida,1);     %define os coeficientes da reta de ajuste 1
Y_curva = polyval(coef_curva, valid_desnorm, error1);      %define a variável y da reta de ajuste
X_curva = valid_desnorm;                                   %define a variável x da reta de ajuste

%Criação da reta modelo
x= 0 : 1;
y = x;

%Grafico de correlacao entre os dados da simulcao e os da base
figure(2)                                   
subplot(2,1,1);                            
plot(valid_desnorm(1,:),saida(1,:),'or');            %gráfico da saída calculada x saída real
hold on                                     %permite que mais de uma curva seja plotada no mesmo gráfico
plot(X_curva,Y_curva,'-k');                 %plota a reta de melhor ajuste dos pontos (reta calculada)
plot(x,y,'--b'); %plota a reta modelo (y = x)title1 = strcat('Microbial Concentration Observed vs Calculated - Train R^2 =', num2str(r2_curve1)); %artificio para plotar texto e valores no título do gráfico
title1 = strcat('Delta P (k+5) [PSI] Observed vs Calculated - R^2 =', num2str(r2_curva_k5)); %artificio para plotar texto e valores no título do gráfico
title(title1,'FontSize',14); %título do primeiro gráfico
xlabel('\bf Delta P Calculated','FontSize',12); %legenda do eixo x, com fonte de tamanho 9
ylabel('\bf Delta P Observed','FontSize',12); %legenda do eixo y, com fonte de tamanho 9
legend('Train','Test', 'Location','northwest'); %legenda do gráfico no canto superior direito%
hold off

%Grafico de Comapracao - sobreposicao dos dados da simulcao e os da base
subplot(2,1,2);                     
plot(saida(1,:),'-k'); %grafico da saída real
hold on
plot(valid_desnorm(1,:),'vm');                              %gráfico da saída calculada
xlabel('\bf Time [s]','FontSize',12); %legenda do eixo x, com fonte de tamanho 9
ylabel('\bf ANN Delta P','FontSize',12); %legenda do eixo y, com fonte de tamanho 9
legend('Samples','ANN', 'Location','northwest'); %legenda do gráfico no canto superior direito%
title('Delta P (k+5) [PSI] Samples vs ANN','FontSize',14); %título do gráfico, com fonte do tamanho 11
hold off

print('-dtiff', '-r600', strcat(title_0,'_k_5',".tiff"))
savefig (strcat(title_0,'_k_5'));          %salva o gráfico como arquivo editável do matlab




%% Funcao para o calcula do R2 - por minimos quadrados
function [r2_erros_diferentes] = calc_r2 (vetor_x, vetor_y)

%Método dos Mínimos Quadrados

%Fórmula de R2 usada quando todos os pontos possuem o mesmo desvio da reta (mesmo erro): 
%R2 = (x - xmedio) * (y - ymedio) / (n - 1)

%Quando os pontos possuem erros diferentes, basta fazer a divisão pelo fator que considera os erros: 
%R2_real = R2 / (desvpad_x * desvpad_y)

n = length(vetor_x);

for k = 1:n
    delta_x(k) = vetor_x(k) - mean(vetor_x); %(x - xmedio) para cada elemento
    delta_y(k) = vetor_y(k) - mean(vetor_y); %(y - ymedio) para cada elemento 
end

%Agora temos dois vetores linha que precisam ser multiplicados. Para que R2 seja um escalar, multiplica-se um vetor linha por um vetor coluna, para que o resultado seja 1x1, fazendo a transposta de y.

r2_erros_iguais = ((delta_x) * (delta_y')) / (n - 1) ; %tem que fazer a transposta de y para gerar um escalar

%Agora para erros diferentes
desvpad_x = std(delta_x);
desvpad_y = std(delta_y);

r2_erros_diferentes = r2_erros_iguais / (desvpad_x * desvpad_y);

end 

