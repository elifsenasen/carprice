namespace CarPriceWebApi.Models
{
  public class EvaluationMetrics
  {
    public double Train_MSE { get; set; }
    public double Train_MAE { get; set; }
    public double Train_R2 { get; set; }
    public double Test_MSE { get; set; }
    public double Test_MAE { get; set; }
    public double Test_R2 { get; set; }
  }

  public class EvaluationResult
    {
        public EvaluationMetrics Results { get; set; }
    }
}