using Microsoft.AspNetCore.Mvc;
using CarPriceWebApi.Models;

[ApiController]
[Route("[controller]")]
public class PredictionController : ControllerBase
{
    [HttpPost]
    public IActionResult Predict([FromBody] CarInput input)
    {
        return Ok(new { predicted_price = 12345 }); //for testing
    }
}
