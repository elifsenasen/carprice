namespace CarPriceWebApi.Models
{
  public class EvaluationMetrics
  {
    public float train_mse { get; set; }
    public float train_mae { get; set; }
    public float train_r2 { get; set; }
    public float test_mse { get; set; }
    public float test_mae { get; set; }
    public float test_r2 { get; set; }
  }

  public class EvaluationResult
    {
        public EvaluationMetrics Results { get; set; }
    }
}