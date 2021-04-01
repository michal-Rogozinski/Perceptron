using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace Perceptron
{
    public static class Program
    {
        static readonly int ITERATION_COUNT = 50;
        static readonly double LAMBDA = 0.1;

        static void Main(string[] args)
        {
            Random rng = new Random(Guid.NewGuid().GetHashCode());
            int seed = rng.Next();
            //Datalist prep
            List<List<double>> data1 = new List<List<double>>();
            List<List<double>> data2 = new List<List<double>>();
            List<List<double>> data3 = new List<List<double>>();
            List<int> labels1 = new List<int>();
            List<int> labels2 = new List<int>();
            List<int> labels3 = new List<int>();
            List<int> res1 = new List<int>();
            List<int> res2 = new List<int>();
            List<int> res3 = new List<int>();
            //Linear order file parsing and shuffling
            parseFile(args[0], data1, labels1,0);
            parseFile(args[0], data2, labels2,1);
            parseFile(args[0], data3, labels3,2);
            Console.WriteLine("**********************END OF DATA PARSE************************");
            Shuffle(data1,seed);
            Shuffle(data2,seed);
            Shuffle(data3,seed);
            Shuffle(labels1,seed);
            Shuffle(labels2,seed);
            Shuffle(labels3,seed);
            /*Console.WriteLine("**********************TABLE 1************************");
            Print2DTable(data1);
            Console.WriteLine("**********************TABLE 2************************");
            Print2DTable(data2);
            Console.WriteLine("**********************TABLE 3************************");
            Print2DTable(data3);*/
            //Computation
            res1 = Neuron(data1,labels1);
            res2 = Neuron(data2,labels2);
            res3 = Neuron(data3,labels3);
            //Output
            Console.WriteLine("**********************FINAL TABLE************************");
            printResult(res1,labels1,0);
            printResult(res2,labels2,1);
            printResult(res3,labels3,2);
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
                        if (split[split.Length - 1].Equals("Iris - virginica"))
                        {
                            for (int i = 0; i < split.Length - 1; i++)
                            {
                                tmp.Add(Double.Parse(split[i], CultureInfo.InvariantCulture));
                            }
                            labelsSet.Add(0);
                        }
                        else if (split[split.Length - 1].Equals("Iris-versicolor"))
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
            weightInitalization(weights, 0.0);

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
        public static List<int> Neuron(List<List<double>> dataSet, List<int> labels)
        {
            List<List<double>> data1 = dataSet;
            List<List<double>> data2 = new List<List<double>>();
            List<int> testLabelsList = new List<int>();
            double[] weight = new double[data1[0].Count];
            double acc = 0.0d;

            for (int i = 0;i < dataSet.Count / 4; i++)
            {
                data2.Add(dataSet[i]);
                testLabelsList.Add(labels[i]);
            }

            weight = setWeights(data2, testLabelsList,ITERATION_COUNT, LAMBDA);

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
        public static void printResult(List<int> list,List<int> checkList,int op)
        {
            if (op == 0)
            {
                Console.WriteLine("Mode 1 - 0 : Iris-setosa | 1 : Iris-versicolor");
                for (int i = 0; i < list.Count; i++)
                {
                    Console.WriteLine("Generated label : " + list[i] + " ,designated label : " + checkList[i]);
                    if (list[i] == 0 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : "+ (i + 1) + " - " + "Iris-setosa");
                    }
                    else if(list[i] == 1 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : " + (i + 1) + " - " + "Iris-versicolor");
                    }
                    else
                    {
                        Console.WriteLine("Failed match");
                    }
                }
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
                    }
                    else if(list[i] == 1 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : " + (i + 1) + " - " + "Iris-virginica");
                    }
                    else
                    {
                        Console.WriteLine("Failed match");
                    }
                }
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
                    }
                    else if(list[i] == 1 && list[i] == checkList[i])
                    {
                        Console.WriteLine("Result at entry : " + (i + 1) + " - " + "Iris - versicolor");
                    }
                    else
                    {
                        Console.WriteLine("Failed match");
                    }
                }
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
        public static int[,] consolidate(List<int> n1,List<int> n2,List<int> n3,int length)
        {
            int[,] output = new int[length, 3];
            for(int i = 0;i < n1.Count; i++)
            {
                output[i, 0] = n1[i];
            }
            for (int i = 0; i < n2.Count; i++)
            {
                output[i, 1] = n1[i];
            }
            for (int i = 0; i < n3.Count; i++)
            {
                output[i, 2] = n1[i];
            }
            return output;
        }
    }
}
