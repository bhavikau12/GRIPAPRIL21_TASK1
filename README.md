{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "GRIP JAN2021",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8C7OpQrmNv0E"
      },
      "source": [
        "# **Author**: Addepalli Divya Ishwarya"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZuAhkxlPVn9J"
      },
      "source": [
        "**Data Science & Business Analytics**\r\n",
        "\r\n",
        "Task 1 : Prediction using Supervised Machine Learning\r\n",
        "\r\n",
        "GRIP - The Sparks Foundation\r\n",
        "\r\n",
        "In this task,the goal is to predict the percentage of a student based on the number of study hours using simple linear regression."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pHn5ZFKp2z55"
      },
      "source": [
        "import pandas as pd\r\n",
        "import matplotlib.pyplot as plt\r\n",
        "import seaborn as sns\r\n",
        "from sklearn.model_selection import train_test_split\r\n",
        "from sklearn.linear_model import LinearRegression\r\n",
        "import sklearn.metrics as metrics"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dlHt-dBmShKe"
      },
      "source": [
        "**Read a comma-separated values (csv) file into DataFrame**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UoqgkV-L6WMG",
        "outputId": "a363ba27-149d-4f58-fc9a-d787dcf14950"
      },
      "source": [
        "data = pd.read_csv(\"http://bit.ly/w-data\");\r\n",
        "print(\"successfully data imported\")"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "successfully data imported\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QjcAz5-RTHm3"
      },
      "source": [
        "**descriptive statistics**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 297
        },
        "id": "cHwUAMQ96gdz",
        "outputId": "12593b61-f816-40f3-c8ac-7616708bb9f6"
      },
      "source": [
        "data.describe()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Hours</th>\n",
              "      <th>Scores</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>25.000000</td>\n",
              "      <td>25.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>5.012000</td>\n",
              "      <td>51.480000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>2.525094</td>\n",
              "      <td>25.286887</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.100000</td>\n",
              "      <td>17.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>2.700000</td>\n",
              "      <td>30.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>4.800000</td>\n",
              "      <td>47.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>7.400000</td>\n",
              "      <td>75.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>9.200000</td>\n",
              "      <td>95.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "           Hours     Scores\n",
              "count  25.000000  25.000000\n",
              "mean    5.012000  51.480000\n",
              "std     2.525094  25.286887\n",
              "min     1.100000  17.000000\n",
              "25%     2.700000  30.000000\n",
              "50%     4.800000  47.000000\n",
              "75%     7.400000  75.000000\n",
              "max     9.200000  95.000000"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        },
        "id": "4Jfmq0JzHfTn",
        "outputId": "53066b14-b1f4-4a58-c6cc-0c3f88c2ef2d"
      },
      "source": [
        "data.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Hours</th>\n",
              "      <th>Scores</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2.5</td>\n",
              "      <td>21</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>5.1</td>\n",
              "      <td>47</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3.2</td>\n",
              "      <td>27</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>8.5</td>\n",
              "      <td>75</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3.5</td>\n",
              "      <td>30</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Hours  Scores\n",
              "0    2.5      21\n",
              "1    5.1      47\n",
              "2    3.2      27\n",
              "3    8.5      75\n",
              "4    3.5      30"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H7HGhfKaLBzO",
        "outputId": "4c8b286c-b653-43a3-9755-5c4e6bb9869c"
      },
      "source": [
        "data.info()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 25 entries, 0 to 24\n",
            "Data columns (total 2 columns):\n",
            " #   Column  Non-Null Count  Dtype  \n",
            "---  ------  --------------  -----  \n",
            " 0   Hours   25 non-null     float64\n",
            " 1   Scores  25 non-null     int64  \n",
            "dtypes: float64(1), int64(1)\n",
            "memory usage: 528.0 bytes\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MJlnJtTPRbR9"
      },
      "source": [
        "**Data Pre-Processing**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MXeVOhbSLOmj",
        "outputId": "14f25a5d-c940-450f-d66c-01dec9e00f1f"
      },
      "source": [
        "x=data['Hours'].values.reshape(-1,1)\r\n",
        "print(x)\r\n",
        "print(x.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[2.5]\n",
            " [5.1]\n",
            " [3.2]\n",
            " [8.5]\n",
            " [3.5]\n",
            " [1.5]\n",
            " [9.2]\n",
            " [5.5]\n",
            " [8.3]\n",
            " [2.7]\n",
            " [7.7]\n",
            " [5.9]\n",
            " [4.5]\n",
            " [3.3]\n",
            " [1.1]\n",
            " [8.9]\n",
            " [2.5]\n",
            " [1.9]\n",
            " [6.1]\n",
            " [7.4]\n",
            " [2.7]\n",
            " [4.8]\n",
            " [3.8]\n",
            " [6.9]\n",
            " [7.8]]\n",
            "(25, 1)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Cwksj_LGM2aC",
        "outputId": "2d894703-8c46-4db0-bc28-22d66ef466b6"
      },
      "source": [
        "y=data['Scores'].values.reshape(-1,1)\r\n",
        "print(y)\r\n",
        "print(y.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[21]\n",
            " [47]\n",
            " [27]\n",
            " [75]\n",
            " [30]\n",
            " [20]\n",
            " [88]\n",
            " [60]\n",
            " [81]\n",
            " [25]\n",
            " [85]\n",
            " [62]\n",
            " [41]\n",
            " [42]\n",
            " [17]\n",
            " [95]\n",
            " [30]\n",
            " [24]\n",
            " [67]\n",
            " [69]\n",
            " [30]\n",
            " [54]\n",
            " [35]\n",
            " [76]\n",
            " [86]]\n",
            "(25, 1)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "V_2one2hUtK7"
      },
      "source": [
        "**Data Visualizing**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "pLLI7S-lNDhx",
        "outputId": "e611cfde-3d3e-493c-e819-46a8c44401a4"
      },
      "source": [
        "data.plot(x='Hours', y='Scores', style='o')\r\n",
        "plt.title(\"Hours vs Percentage\")\r\n",
        "plt.xlabel(\"Hours Studied\")\r\n",
        "plt.ylabel(\"Percentage Score\")\r\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S4KEW7q8VEUr"
      },
      "source": [
        "**Defining and Training the model**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z0rWUbqiPU7H",
        "outputId": "0d6eca31-7bfb-4300-9576-eeb85a916277"
      },
      "source": [
        "x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)\r\n",
        "model=LinearRegression()\r\n",
        "model.fit(x_train,y_train)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "g6s3VCdqSEpF"
      },
      "source": [
        "prediction=model.predict(x_test)\r\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "X8U54CZ5SVwI",
        "outputId": "754b90e7-65fd-4c01-bca3-02dd5cc7ade0"
      },
      "source": [
        "plt.scatter(x_train, y_train,color='steelblue')\r\n",
        "plt.plot(x_train, model.predict(x_train), color = 'lime')\r\n",
        "plt.title('Hours vs Scores during Training set')\r\n",
        "plt.xlabel('No of Hours Studied')\r\n",
        "plt.ylabel('Scores')\r\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dh0fCRExXamr"
      },
      "source": [
        "**Prediction Metrics**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kWvAtp4JUITa",
        "outputId": "e55e4371-fa22-49a2-e7db-53059d8a316a"
      },
      "source": [
        "print(\"MAE {}\".format(metrics.mean_absolute_error(y_test,prediction)))\r\n",
        "print(\"MSE {}\".format(metrics.mean_squared_error(y_test,prediction)))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "MAE 4.130879918502482\n",
            "MSE 20.33292367497996\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "EBb3MKciUobE",
        "outputId": "09550810-2711-42da-d816-b9e82887c72d"
      },
      "source": [
        "plt.scatter(x_test, y_test,color='blue')\r\n",
        "plt.plot(x_test, model.predict(x_test), color = 'green')\r\n",
        "plt.title('Predicted : Hours vs Scores')\r\n",
        "plt.xlabel('Hours Studied')\r\n",
        "plt.ylabel('Scores')\r\n",
        "plt.show()\r\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GZs_m249XoQa"
      },
      "source": [
        "**Actual vs predicted**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aYGlzrR1UxjQ",
        "outputId": "74d39f78-4693-40f9-c935-c03b9ee528b6"
      },
      "source": [
        "result=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':prediction.flatten()})\r\n",
        "print(result)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "   Actual  Predicted\n",
            "0      20  16.844722\n",
            "1      27  33.745575\n",
            "2      69  75.500624\n",
            "3      30  26.786400\n",
            "4      62  60.588106\n",
            "5      35  39.710582\n",
            "6      24  20.821393\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yZHFvZxgX2fT"
      },
      "source": [
        "\r\n",
        "**Q. What will be the predicted score if a student studies for 8hr?**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1i51UfnmU8Lx",
        "outputId": "61191fc0-b0f5-430c-8eee-30ce09d36b38"
      },
      "source": [
        "print('If a student studies for 8hr , then he may score {}'.format(model.predict([[8]])))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "If a student studies for 8hr , then he may score [[81.46563097]]\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
} 
