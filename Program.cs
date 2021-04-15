using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace Perceptron
{
    public static class Program
    {

        static void Main(string[] args)
        {
            int ITERATION_COUNT = 1;
            double LAMBDA = Double.Parse(args[2], CultureInfo.InvariantCulture);

            Random rng = new Random(Guid.NewGuid().GetHashCode());
            int seed = rng.Next();

            //Datalist prep
            List<List<double>> data1 = new List<List<double>>();
            List<List<double>> tData = new List<List<double>>();
            List<int> labels1 = new List<int>();
            List<int> tLabels = new List<int>();
            List<int> res1 = new List<int>();

            //Linear order file parsing and shuffling. Last integer stands for file mode - 0 is setosa and versicolor,1 is setosa and virginica,2 is virginica and versicolor

            parseFile(args[0], data1, labels1,0);
            parseFile(args[1], tData, tLabels,0);

            double[] weightsList = new double[tData[0].Count];
            Shuffle(tData, seed);
            Shuffle(tLabels, seed);
            weightsList = setWeights(tData, tLabels, ITERATION_COUNT, LAMBDA);

            Console.WriteLine("**********************END OF DATA PARSE************************");

            Shuffle(data1,seed);
            Shuffle(labels1,seed);

            //Computation
            res1 = Neuron(data1,labels1,weightsList);

            //Output
            Console.WriteLine("**********************FINAL TABLE************************");
            printResult(res1,labels1,0);

            //Manual testing
            Console.WriteLine("*********************************************************");
            double[] manual = new double[weightsList.Length];
            Boolean runnning = true;
            while (runnning)
            {
                Console.WriteLine("Enter data for manual testing. Current length of vector is : " + weightsList.Length);
                for (int i = 0; i < weightsList.Length; i++)
                {
                    Console.WriteLine("Enter parameter no. " + i);
                    manual[i] = Double.Parse(Console.ReadLine(), CultureInfo.InvariantCulture);
                }
                ManualNeuron(manual, weightsList, 0);
                Console.WriteLine("Do you want to continue ?");
                string choice = Console.ReadLine();
                if(choice.Equals("yes") || choice.Equals("y"))
                {
                    continue;
                }
                if (choice.Equals("no") || choice.Equals("n"))
                {
                    runnning = false;
                }
            }
        }
        public static void parseFile(String path,List<List<double>> dataSet,List<int> labelsSet,int splitOpt)
        {
            StreamReader reader = new StreamReader(path);
            string lineIn;
            try
            {
                while ((lineIn = reader.ReadLine()) is not null)
                {
                    string[] split = lineIn.Split(';');
                    List<double> tmp = new List<double>();
                    if(splitOpt == 0)
                    {
                        if(split[split.Length - 1].Equals("Iris-setosa"))
                        {
                            for(int i = 0;i < split.Length -1;i++)
                            {
                                tmp.Add(Double.Parse(split[i],CultureInfo.InvariantCulture));
                            }
                            labelsSet.Add(0);
                        }
                        else if(split[split.Length - 1].Equals("Iris-versicolor"))
                        {
                            for (int i = 0; i < split.Length - 1; i++)
                            {
                                tmp.Add(Double.Parse(split[i], CultureInfo.InvariantCulture));
                            }
                            labelsSet.Add(1);
                        }
                        else
                        {
                            continue;
                        }
                    }
                    if (splitOpt == 1)
                    {
                        if (split[split.Length - 1].Equals("Iris-setosa"))
                        {
                            for (int i = 0; i < split.Length - 1; i++)
                            {
                                tmp.Add(Double.Parse(split[i], CultureInfo.InvariantCulture));
                            }
                            labelsSet.Add(0);
                        }
                        else if (split[split.Length - 1].Equals("Iris-virginica"))
                        {
                            for (int i = 0; i < split.Length - 1; i++)
                            {
                                tmp.Add(Double.Parse(split[i], CultureInfo.InvariantCulture));
                            }
                            labelsSet.Add(1);
                        }
                        else
                        {
                            continue;
                        }
                    }
                    if (splitOpt == 2)
                    {
                        if (split[split.Length - 1].Equals("Iris-versicolor"))
                        {
                            for (int i = 0; i < split.Length - 1; i++)
                            {
                                tmp.Add(Double.Parse(split[i], CultureInfo.InvariantCulture));
                            }
                            labelsSet.Add(0);
                        }
                        else if (split[split.Length - 1].Equals("Iris-virginica"))
                        {
                            for (int i = 0; i < split.Length - 1; i++)
                            {
                                tmp.Add(Double.Parse(split[i], CultureInfo.InvariantCulture));
                            }
                            labelsSet.Add(1);
                        }
                        else
                        {
                            continue;
                        }
                    }
                    dataSet.Add(tmp);
                }
            }
            catch (IOException)
            {
                Console.WriteLine("IO error");
            }
        }
        public static void Print2DTable(List<List<double>> set)
        {
            for(int i = 0;i < set.Count; i++)
            {
                for(int j = 0;j < set[0].Count - 1; j++)
                {
                    Console.WriteLine(set[i][0].ToString() + " " + set[i][1].ToString() + " " + set[i][2].ToString() + " " + set[i][3].ToString());
                }
            }
        }
        public static void PrintTable(List<int> set)
        {
            foreach(int i in set)
            {
                Console.WriteLine("Label: " + i);
            }
        }
        private static void weightInitalization(double[] weight,double value)
        {
            for(int i = 0;i < weight.Length;i++)
            {
                weight[i] = value;
            }
        }
        private static int sign(double i)
        {
            return i > 0 ? 1 : 0;
        }
        public static double[] setWeights(List<List<double>> dataSet,List<int> labels,int iterNum,double lambda)
        {
            int attNum;
            int recNum;
            if (dataSet is null)
            {
                return null;
            }
            else
            {
                recNum = dataSet.Count;
                attNum = dataSet[0].Count;
            }
            double[] weights = new double[attNum];
            double y;
            weightInitalization(weights, 1.0);

            for(int i = 0;i < iterNum; i++)
            {
                for(int m = 0;m < recNum; m++)
                {
                    y = 0;
                    for(int j = 0;j < attNum; j++)
                    {
                        y += weights[j] * dataSet[m][j];
                    }
                    y = sign(y);
                    for (int k = 0; k < attNum; k++)
                    {
                        weights[k] = weights[k] + lambda * (labels[m] - y) * dataSet[m][k];
                }
                }
            }
            return weights;
        }
        public static List<int> Neuron(List<List<double>> dataSet, List<int> labels,double[] weightList)
        {
            List<List<double>> data1 = dataSet;
            double[] weight = weightList;
            double acc = 0.0d;

            string weights = null;
            foreach(double i in weight)
            {
                weights += " " + String.Format("{0:0.##}", i);
            }
            Console.WriteLine("Weights for neuron : " + weights);

            int recordNum = data1.Count;
            double y;

            List<int> outLabels = new List<int>();

            for(int i =0;i < recordNum; i++)
            {
                y = 0;
                for(int j = 0;j < weight.Length; j++)
                {
                    y += weight[j] * data1[i][j];
                }
                outLabels.Add(sign(y));
            }

            for(int k = 0;k < recordNum; k++)
            {
                if(labels[k] == outLabels[k])
                {
                    acc++;
                }
            }

            Console.WriteLine("Neural Accuracy : " + String.Format("{0:0.##}",(acc/recordNum)));

            return outLabels;
        }
        public static void ManualNeuron(double[] list,double[] weightList,int op)
        {
            double y = 0;
            for(int j = 0;j < weightList.Length; j++)
            {
                y += weightList[j] * list[j];
            }
            y = sign(y);
            if(op == 0)
            {
                if(y == 0)
                {
                    Console.WriteLine("Manual input result : " + sign(y) + " Iris-setosa");
                }
                if(y == 1)
                {
                    Console.WriteLine("Manual input result : " + sign(y) + " Iris-versicolor");

                }
            }
            if (op == 1)
            {
                if (y == 0)
                {
                    Console.WriteLine("Manual input result : " + sign(y) + " Iris-setosa");
                }
                if (y == 1)
                {
                    Console.WriteLine("Manual input result : " + sign(y) + " Iris-virginica");

                }
            }
            if (op == 2)
            {
                if (y == 0)
                {
                    Console.WriteLine("Manual input result : " + sign(y) + " Iris-versicolor");
                }
                if (y == 1)
                {
                    Console.WriteLine("Manual input result : " + sign(y) + " Iris-virginica");

                }
            }
        }
        public static void printResult(List<int> list,List<int> checkList,int op)
        {
            double res1 = 0.0d;
            double res2 = 0.0d;
            if (op == 0)
            {
                Console.WriteLine("Mode 1 - 0 : Iris-setosa | 1 : Iris-versicolor");
                for (int i = 0; i < list.Count; i++)
                {
                    Console.WriteLine("Generated label : " + list[i] + " ,designated label : " + checkList[i]);
                    if (list[i] == 0 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : "+ (i + 1) + " - " + "Iris-setosa");
                        res1++;
                    }
                    else if(list[i] == 1 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : " + (i + 1) + " - " + "Iris-versicolor");
                        res2++;
                    }
                    else
                    {
                        Console.WriteLine("Failed match");
                    }
                }
                Console.WriteLine("Accuracy for Iris-setosa : " + (double)(res1 / list.Count) + " ,accuracy for Iris-versicolor : " + (double)(res2 / list.Count));
            }
            if (op == 1)
            {
                Console.WriteLine("Mode 2 - 0 : Iris-setosa | 1 : Iris-virginica");
                for (int i = 0; i < list.Count; i++)
                {
                    Console.WriteLine("Generated label : " + list[i] + " ,designated label : " + checkList[i]);
                    if (list[i] == 0 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : " + (i+1) + " - " + "Iris-setosa");
                        res1++;
                    }
                    else if(list[i] == 1 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : " + (i + 1) + " - " + "Iris-virginica");
                        res2++;
                    }
                    else
                    {
                        Console.WriteLine("Failed match");
                    }
                }
                Console.WriteLine("Accuracy for Iris-setosa : " + (double)(res1 / list.Count) + " ,accuracy for Iris-virginica : " + (double)(res2 / list.Count));
            }
            if (op == 2)
            {
                Console.WriteLine("Mode 3 - 0 : Iris-virginica | 1 : Iris - versicolor");
                for (int i = 0; i < list.Count; i++)
                {
                    Console.WriteLine("Generated label : " + list[i] + " ,designated label : " + checkList[i]);
                    if (list[i] == 0 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : " + (i + 1) + " - " + "Iris-virginica");
                        res1++;
                    }
                    else if(list[i] == 1 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : " + (i + 1) + " - " + "Iris-versicolor");
                        res2++;
                    }
                    else
                    {
                        Console.WriteLine("Failed match");
                    }
                }
                Console.WriteLine("Accuracy for Iris-virginica : " + (double)(res1 / list.Count) + " ,accuracy for Iris-versicolor : " + (double)(res2 / list.Count));
            }
        }
        public static void Shuffle<T>(IList<T> list,int seed)
        {
            Random rng = new Random(seed);
            int n = list.Count;
            while(n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
