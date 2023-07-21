/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.trabalhoinvestimentoia;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.functions.SMO;
/**
 *
 * @author wilson
 */
public class TrabalhoInvestimentoIASVM {

    public static void main(String[] args) throws Exception {
        try {
            DataSource caminho = new DataSource("/media/wilson/SD_CARD/TCC/Analise_de_Investimento_IA/trabalhoInvestimentoIA/Variação_Petrobras.csv");
            Instances arquivo = caminho.getDataSet();
            int indexAtributoPrever = arquivo.attribute("Fechamento").index();
            arquivo.setClassIndex(indexAtributoPrever);
            SMO svm = new SMO();
            svm.buildClassifier(arquivo);
            double[] valorFuturoInstancia = arquivo.lastInstance().toDoubleArray();
            double pcp = svm.classifyInstance(arquivo.lastInstance());
            
            System.out.println("O comportamento do grafico será " + pcp);
            
        } catch (Exception e) {
            System.out.println("Erro " + e);
        }
        
    }
}
/*
Erro: Não foi possível localizar nem carregar a classe principal de.akquinet.ats.ak40.weka.WekaJ48DemoStarter
Causada por: java.lang.ClassNotFoundException: de.akquinet.ats.ak40.weka.WekaJ48DemoStarter
*/
