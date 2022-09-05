%% 
close all
clear all
clc

%% Carregamento da Planilha de dados para treino
M = load('Treino_div_TrabFinal_MQI2104.dat');      %carrega o arquivo de dados%
M = M';                                     %calcula a matriz transposta
entrada = M(2:7,:);                         %define os dados de entrada da rede
saida = M(8,:);                             %define os dados de sa�da da rede
                            % Para o k+1 linha 8 da matriz M
                            % Para o k+5 linha 12 da matriz M
%% Normaliza��o da Planilha de dados
%define os par�metros m�ximos e m�nimos das matrizes de entrada e sa�da e faz normaliza��o
[entradan,minentrada,maxentrada,saidan,minsaida,maxsaida]=premnmx(entrada,saida);

%% Carregamento e normalizacao da Planilha de dados para treino
N = load('Teste_div_TrabFinal_MQI2104.dat'); %carrega arquivo de dados para teste%
N = N';
ent_teste = N(2:7,:); %dados de entrada da rede%
saida_teste = N(8,:); %dados de saida da rede%
                            % Para o k+1 linha 8 da matriz M
                            % Para o k+5 linha 12 da matriz M
[ent_teste_norm] = tramnmx(ent_teste,minentrada,maxentrada); %normaliza dados%

%% Definicao da Topologia da Rede
net = newff(minmax(entradan(:,:)),[8,1],{'tansig','purelin'},'trainlm');
%=tipo_de_rede(limite_ent,[#neuronios nas CIs e CS],{func de ativacao},alg de treinamento)

%% Treinamento da rede
% Parametros de treinamento
net.trainParam.epochs = 1500;               %n� maximo de iteracoes
net.trainParam.time = 600;                  %tempo maximo
net.trainParam.goal = 1e-4;                 %converg�ncia desejada%
net.performFcn = 'sse';                     %fun��o objetivo a ser minimizada%
net.trainParam.min_grad = 1e-100;           %m�nimo gradiente%
%net.trainParam.mu_max = 1e+400;            %m�ximo MU%

% Inicializacao do treinamento
net.initFcn = 'initlay';                            %fun��o que inicia os pesos e bias%
net = init(net);
[net,tr] = train(net,entradan(:,:),saidan(:,:));    %realiza o treinamento da rede%
                                                    %pesos e bias da rede determinados e guardados em 'net'%

% Armazenamento dos par�metros e salva a rede 
IW_1 = net.IW{1};           % Pesos da camada de entrada, usa-se IW. Comunicacao entre CE e CI_ini
b_1 = net.b{1};             % bias da omunicacao entre CE e CI_ini
LW_2 = net.LW{2};           % LW � usado nas camadas seguinte; Comunicacao entre as CIi ou da CI_final e CS
b_2 = net.b{2};             % bias da comunicacao entre as CIi ou da CI_final e CS

save 'FF ANN MQI2104' net;           %salva a rede, com todos os parametros e vari�veis do teste que foi rodado

%% Simula��o de teste
treino_norm = sim(net,entradan(:,:));                 %simula com os dados de entrada do
treino_desnorm = postmnmx(treino_norm,minsaida,maxsaida);          %desnormaliza dados de saida%

teste_norm = sim(net,ent_teste_norm(:,:));               %simula com os dados de entrada do arquivo de teste%
teste_desnorm = postmnmx(teste_norm,minsaida,maxsaida); %desnormaliza dados de saida%

%% Graficos do Treinamento
%Cria��o da reta de ajuste linear para o treinamento
[coef_curva, error1] = polyfit(treino_desnorm,saida,1);     %define os coeficientes da reta de ajuste 1
Y_curva = polyval(coef_curva, treino_desnorm, error1);      %define a vari�vel y da reta de ajuste
X_curva = treino_desnorm;                                   %define a vari�vel x da reta de ajuste

%Cria��o da reta modelo
x= 0 : 1;
y = x;

%C�lculo do R2 e dos coeficientes de erro da reta 1
r2_curva = calc_r2(treino_desnorm,saida);
sse_curva = sse(net, saida, treino_desnorm);
mse_curva = immse(saida, treino_desnorm);
rmse_curva = sqrt(mse_curva);

%Grafico de correlacao entre os dados da simulcao e os da base de treino
figure(1)                                   %abre uma janela para plotar a imagem
subplot(2,2,1);                             %divide a janela de imagem em 4 quadrantes: 2 na horizontal e 2 na vertical,
                                            %e plota no quadrante a esquerda no alto
plot(treino_desnorm,saida,'or');            %gr�fico da sa�da calculada x sa�da real
hold on                                     %permite que mais de uma curva seja plotada no mesmo gr�fico
plot(X_curva,Y_curva,'-k');                 %plota a reta de melhor ajuste dos pontos (reta calculada)
plot(x,y,'--b'); %plota a reta modelo (y = x)title1 = strcat('Microbial Concentration Observed vs Calculated - Train R^2 =', num2str(r2_curve1)); %artificio para plotar texto e valores no t�tulo do gr�fico
title1 = strcat('Delta P (psi) Observed vs Calculated - Train   R^2 =', num2str(r2_curva)); %artificio para plotar texto e valores no t�tulo do gr�fico
title(title1,'FontSize',11); %t�tulo do primeiro gr�fico
xlabel('Delta P Calculated','FontSize',9); %legenda do eixo x, com fonte de tamanho 9
ylabel('Delta P Observed','FontSize',9); %legenda do eixo y, com fonte de tamanho 9
hold off                            %encerra os plot's no mesmo gr�fico

%Grafico de Comapracao - sobreposicao dos dados da simulcao e os da base de treino
subplot(2,2,2);                     %divide a janela de imagem em 4 quadrantes: 2 na horizontal e 2 na vertical,
                                    %e plota no quadrante a direita no alto
plot(saida,'-k'); %grafico da sa�da real
hold on
plot(treino_desnorm,'vm');                              %gr�fico da sa�da calculada
xlabel('Samples','FontSize',9);                        %legenda do eixo x, com fonte de tamanho 9
ylabel('ANN Delta P','FontSize',9);                  %legenda do eixo y, com fonte de tamanho 9
legend('Samples','ANN', 'Location','southeast');   %legenda do gr�fico no canto superior direito%
%legend('boxoff')                                       %retira a caixa em que as legendas ficam
title('Delta P (psi) Samples vs ANN - Train','FontSize',11); %t�tulo do gr�fico, com fonte do tamanho 11
hold off

%% Graficos do Teste
%Cria��o da reta de ajuste linear para o treinamento
[coef_curva, error1] = polyfit(teste_desnorm,saida_teste,1);     %define os coeficientes da reta de ajuste 1
Y_curva = polyval(coef_curva, teste_desnorm, error1);      %define a vari�vel y da reta de ajuste
X_curva = teste_desnorm;                                   %define a vari�vel x da reta de ajuste

%Cria��o da reta modelo
x= 0 : 1;
y = x;

%C�lculo do R2 e dos coeficientes de erro da reta 1
r2_curva = calc_r2(teste_desnorm,saida_teste);
sse_curva = sse(net, saida_teste, teste_desnorm);
mse_curva = immse(saida_teste, teste_desnorm);
rmse_curva = sqrt(mse_curva);

%Grafico de correlacao entre os dados da simulcao e os da base de treino
figure(1)                                   %abre uma janela para plotar a imagem
subplot(2,2,3);                             %divide a janela de imagem em 4 quadrantes: 2 na horizontal e 2 na vertical,
                                            %e plota no quadrante a esquerda embaixo
plot(teste_desnorm,saida_teste,'or');            %gr�fico da sa�da calculada x sa�da real
hold on                                     %permite que mais de uma curva seja plotada no mesmo gr�fico
plot(X_curva,Y_curva,'-k');               %plota a reta de melhor ajuste dos pontos (reta calculada)
plot(x,y,'--b'); %plota a reta modelo (y = x)title1 = strcat('Microbial Concentration Observed vs Calculated - Train R^2 =', num2str(r2_curve1)); %artificio para plotar texto e valores no t�tulo do gr�fico
title1 = strcat('Delta P (psi) Observed vs Calculated - Test   R^2 =', num2str(r2_curva)); %artificio para plotar texto e valores no t�tulo do gr�fico
title(title1,'FontSize',11); %t�tulo do primeiro gr�fico
xlabel('Delta P Calculated','FontSize',9); %legenda do eixo x, com fonte de tamanho 9
ylabel('Delta P Observed','FontSize',9); %legenda do eixo y, com fonte de tamanho 9
hold off                            %encerra os plot's no mesmo gr�fico

%Grafico de Comapracao - sobreposicao dos dados da simulcao e os da base de treino
subplot(2,2,4);                     %divide a janela de imagem em 4 quadrantes: 2 na horizontal e 2 na vertical,
                                    %e plota no quadrante a direita embaixo
plot(saida_teste,'-k'); %grafico da sa�da real
hold on
plot(teste_desnorm,'vm'); %gr�fico da sa�da calculada
xlabel('Samples','FontSize',9); %legenda do eixo x, com fonte de tamanho 9
ylabel('ANN Delta P','FontSize',9); %legenda do eixo y, com fonte de tamanho 9
legend('Samples','ANN', 'Location','southeast'); %legenda do gr�fico no canto superior direito%
title('Delta P (psi) Samples vs ANN - Test','FontSize',11); %t�tulo do gr�fico, com fonte do tamanho 11
hold off

%% Funcao para o calcula do R2 - SSE e TSS
function [r2_erros] = calc_r2 (y_exp, y_pred)

    sse_t1 = sum((y_exp - y_pred).^2);

    tss_t1= sum((y_exp - mean(y_exp)).^2);

    r2_erros = 1-(sse_t1/tss_t1);

end 

