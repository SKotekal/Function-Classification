import java.util.*;
import java.io.*;

public class func 
{
	public static void main(String[] args) throws IOException
	{
		Scanner scan = new Scanner(System.in);
		System.out.print("Input absolute path to dataset (.csv): ");
		String path = scan.nextLine();
		
		Scanner fin = new Scanner(new File(path));
		ArrayList<Double> X = new ArrayList<Double>();
		ArrayList<Double> Y = new ArrayList<Double>();
		
		while(fin.hasNext())
		{
			String[] coords = fin.nextLine().split(",");
			X.add(Double.parseDouble(coords[0]));
			Y.add(Double.parseDouble(coords[1]));
		}
		System.out.println(X);
		System.out.println(Y);

		ArrayList<Double> theta = new ArrayList<Double>();
		ArrayList<Double> coeff = new ArrayList<Double>();
		do {
			theta.add(0.0);
			coeff = gradDescent(theta, X, Y, 0.000001, 0.001);
		} while(coeff.get(0) == Double.MAX_VALUE);

		System.out.println(coeff);
	}

	public static ArrayList<Double> gradDescent(ArrayList<Double> theta, ArrayList<Double> X, ArrayList<Double> Y, double err, double alpha)
	{
		ArrayList<Double> ret = new ArrayList<Double>();

		ArrayList<Double> cost_hist = new ArrayList<Double>();
		double J = Double.MAX_VALUE;
		cost_hist.add(0.0);
		cost_hist.add(J);

		int m = X.size();
		int n = theta.size();

		for(int i = 0; i < n; i++)
		{
			ret.add(theta.get(i));
		}

		while(J > err)
		{
			ArrayList<Double> temp = new ArrayList<Double>();
			for(int j = 0; j < n; j++)
			{
				temp.add(ret.get(j));
				double theta_j = ret.get(j);
				for(int i = 0; i < m; i++)
				{
					theta_j -= (alpha/m)*(pderiv(j, ret, X, Y));
				}
				temp.set(j, theta_j);
			}
			for(int j = 0; j < n; j++)
			{
				ret.set(j, temp.get(j));
			}

			J = cost(ret, X, Y);
			cost_hist.add(J);

			if(cost_hist.get(cost_hist.size()-1)-cost_hist.get(cost_hist.size()-2) > err)
			{
				ret.set(0, Double.MAX_VALUE);
				break;
			}
		}

		return ret;
	}

	public static double cost(ArrayList<Double> theta, ArrayList<Double> X, ArrayList<Double> Y)
	{
		double J = 0;
		int m = X.size();
		for(int i = 0; i < m; i++)
		{
			J += Math.pow(polyEval(theta, X.get(i))-Y.get(i), 2);
		}
		J /= (2*((double) m));

		return J;
	}
	public static double polyEval(ArrayList<Double> theta, double x)
	{
		double eval = theta.get(0);
		for(int i = 1; i < theta.size(); i++)
		{
			eval += theta.get(i)*Math.pow(x, i);
		}
		return eval;
	}
	public static double pderiv(int j, ArrayList<Double> theta, ArrayList<Double> X, ArrayList<Double> Y)
	{
		double delta = 0;
		int m = X.size();
		if(j == 0)
		{
			for(int i = 1; i < m; i++)
			{
				delta += (polyEval(theta, X.get(i))-Y.get(i));
			}
		}
		else
		{
			for(int i = 1; i < m; i++)
			{
				delta += (polyEval(theta, X.get(i))-Y.get(i))*Math.pow(X.get(i), j);
			}
		}

		return delta/((double) m);
	}
}
