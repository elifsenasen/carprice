namespace CarPriceWebApi.Models
{
  public class EvaluationMetricsResponse
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
        public EvaluationMetricsResponse Results { get; set; }
    }
}