# Machine learning for sports betting: Should model selection be based on accuracy or calibration?

## Overview

This repository contains the code used to run the betting experiments in our paper "Machine learning for sports betting: Should model selection be based on accuracy or calibration?", published in Machine Learning with Applications ([https://doi.org/10.1016/j.mlwa.2024.100539](https://www.sciencedirect.com/science/article/pii/S266682702400015X?via%3Dihub)). The implementation provided here enables reproducibility of our experiments and offers a foundation for further exploration in this domain.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/conorwalsh99/ml-for-sports-betting.git
   cd ml_for_sports_betting
    ```
    
2. **Install dependencies**
    ```
    poetry install
    ```
    
3. **Run betting experiment pipeline**
    ```
    poetry run python src/run_pipeline.py
    ```

Note: The pipeline run can take some time to complete.

## Citation
If you use this code in your research or any related work, please cite our paper:
```
@article{walsh2024machine,
  title={Machine learning for sports betting: should model selection be based on accuracy or calibration?},
  author={Walsh, Conor and Joshi, Alok},
  journal={Machine Learning with Applications},
  volume={16},
  pages={100539},
  year={2024},
  publisher={Elsevier}
}
```

## License

This repository is open-sourced and free to use, modify, and distribute for any purpose as long as you cite the paper. See the LICENSE file for complete details.

## Acknowledgements

We appreciate your interest in our work. For any questions, feedback, or contributions, please feel free to contact the corresponding author directly at conorwalsh206@gmail.com.