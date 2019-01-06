import weka.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.*;
import java.io.FileReader;

import javafx.application.Application;
import javafx.collections.*;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.stage.Stage;
import javafx.scene.layout.StackPane;
import javafx.scene.chart.XYChart.Series;
import javafx.stage.Stage;

public class A4_ML_Spiral extends Application {

	public static void main(String[] args) throws Exception {
		launch(args);
	}
	
   @Override
   public void start(Stage stage) throws Exception {
		
		FileReader reader = new FileReader("C:\\Users\\ander\\source\\repos\\AHV\\Courses\\LNU\\Web Intelligence\\Assignment4\\spiral.arff");
		Instances instances = new Instances(reader);
		instances.setClassIndex(instances.numAttributes() - 1);
		
		NumberAxis xAxis = new NumberAxis();
		NumberAxis yAxis = new NumberAxis();
		ScatterChart<Number, Number> sc = new ScatterChart<>(xAxis, yAxis);
        Series<Number, Number> as = new Series<>();
		Series<Number, Number> bs = new Series<>();
		Series<Number, Number> cs = new Series<>();
		
		as.setName("A");
		bs.setName("B");
		cs.setName("C");
		
		for (int i = 0; i < instances.numInstances(); i++) {
			Series<Number, Number> s;
			Instance instance = instances.instance(i);
			switch ((int)instance.value(2)) {
			case 0:
				s = as;
				break;
			case 1:
				s = bs;
				break;
			default:
				s = cs;
				break;
			}
			s.getData().add(new XYChart.Data<Number, Number>(instance.value(0), instance.value(1)));
		}
		
		ObservableList<XYChart.Series<Number, Number>> data = FXCollections.observableArrayList();
		data.addAll(as, bs, cs);
		sc.setData(data);
		
	    stage.setTitle("Spiral Scatter Chart");
	    StackPane pane = new StackPane();
	    pane.getChildren().add(sc);
	    stage.setScene(new Scene(pane, 400, 200));
	    stage.show();	    	    
	    
		Logistic log = new Logistic();
		log.setOptions(Utils.splitOptions("-R 1.0E-8 -M -1 -num-decimal-places 4"));
		log.buildClassifier(instances);
		
		Evaluation eval1 = new Evaluation(instances);	
		eval1.evaluateModel(log, instances);		
		
	    System.out.println(eval1.toSummaryString());
	    System.out.println(eval1.toMatrixString("Confusion matrix:"));
		
		MultilayerPerceptron mlp = new MultilayerPerceptron();		
		mlp.setOptions(Utils.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 72"));		
		mlp.buildClassifier(instances);
		
		Evaluation eval2 = new Evaluation(instances);		
		eval2.evaluateModel(mlp, instances);
		
	    System.out.println(eval2.toSummaryString());
	    System.out.println(eval2.toMatrixString("Confusion matrix:"));
	}
}
