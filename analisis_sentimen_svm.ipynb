{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "analisis_sentimen_svm.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyMJveYg6lbnGVqlZ2vJqObM",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Rodiyaturrahmi/analisis_sentimen_svm/blob/main/analisis_sentimen_svm.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import re\n",
        "import nltk\n",
        "\n",
        "from nltk.tokenize import word_tokenize\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import PorterStemmer\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #import Indonesian Stemmer\n",
        "\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.svm import LinearSVC\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from sklearn.metrics import classification_report"
      ],
      "metadata": {
        "id": "WFk2RhVDOl-t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install Sastrawi"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2y8uuhttltRq",
        "outputId": "09711595-aea4-45cb-8c96-b48c2c6d87c5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting Sastrawi\n",
            "  Downloading Sastrawi-1.0.1-py2.py3-none-any.whl (209 kB)\n",
            "\u001b[K     |████████████████████████████████| 209 kB 8.1 MB/s \n",
            "\u001b[?25hInstalling collected packages: Sastrawi\n",
            "Successfully installed Sastrawi-1.0.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Dataset"
      ],
      "metadata": {
        "id": "e9fztdjWd_ee"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CZSOInccSCQ3",
        "outputId": "c4dd2315-b724-4d5f-805d-507c91e937b1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dataset = 'drive/MyDrive/sentimen/dataset_tweet_sentimen.csv'\n",
        "data = pd.read_csv(dataset)\n",
        "data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "-U8MvzYBeezj",
        "outputId": "518770b4-b5da-42f2-ad7d-174f15d621b8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                 tweet Sentimen\n",
              "0     https://bit.ly/2L6CcbB  | Seharusnya saat ini...  Negatif\n",
              "1    Klo kata gw Pemerintah tuh lagi menerapkan Her...  Negatif\n",
              "2    Saat ini yang bisa saya lakukan hanya menyiapk...  Negatif\n",
              "3    emosi bgt, lg kondisi begini gue disuruh liput...  Negatif\n",
              "4    Bang gimna pemerintah mau peduli RRC urus ABK ...  Negatif\n",
              "..                                                 ...      ...\n",
              "995  Kalian tak percaya Covid ada, himbauan pemerin...  Positif\n",
              "996  Berat perjuangan pemerintah Aceh, ketika rakya...  Positif\n",
              "997  Anjuran pemerintah tidak boleh berpegangan tan...  Positif\n",
              "998  gue bingung data covid naik terus, yang salah ...  Positif\n",
              "999  Beneran kalo sakit atau penyakit gw kambuh seb...  Negatif\n",
              "\n",
              "[1000 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-44d97f11-d17c-4056-ad19-e88f9f4432d8\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>tweet</th>\n",
              "      <th>Sentimen</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>https://bit.ly/2L6CcbB  | Seharusnya saat ini...</td>\n",
              "      <td>Negatif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Klo kata gw Pemerintah tuh lagi menerapkan Her...</td>\n",
              "      <td>Negatif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Saat ini yang bisa saya lakukan hanya menyiapk...</td>\n",
              "      <td>Negatif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>emosi bgt, lg kondisi begini gue disuruh liput...</td>\n",
              "      <td>Negatif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Bang gimna pemerintah mau peduli RRC urus ABK ...</td>\n",
              "      <td>Negatif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>995</th>\n",
              "      <td>Kalian tak percaya Covid ada, himbauan pemerin...</td>\n",
              "      <td>Positif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>996</th>\n",
              "      <td>Berat perjuangan pemerintah Aceh, ketika rakya...</td>\n",
              "      <td>Positif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>997</th>\n",
              "      <td>Anjuran pemerintah tidak boleh berpegangan tan...</td>\n",
              "      <td>Positif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>998</th>\n",
              "      <td>gue bingung data covid naik terus, yang salah ...</td>\n",
              "      <td>Positif</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>999</th>\n",
              "      <td>Beneran kalo sakit atau penyakit gw kambuh seb...</td>\n",
              "      <td>Negatif</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1000 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-44d97f11-d17c-4056-ad19-e88f9f4432d8')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-44d97f11-d17c-4056-ad19-e88f9f4432d8 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-44d97f11-d17c-4056-ad19-e88f9f4432d8');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "nltk.download('stopwords')\n",
        "nltk.download('punkt')\n",
        "nltk.download('wordnet')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L-iobEBwfsuH",
        "outputId": "63ba66ef-21b5-4116-a958-1e3e24e51b89"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n",
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Unzipping tokenizers/punkt.zip.\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['tweet'][1]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 70
        },
        "id": "E702aC0ngsV5",
        "outputId": "b40bc436-f3e2-421f-de95-5e9222022c34"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'Klo kata gw Pemerintah tuh lagi menerapkan Herd immunity karena kemampuan pemerintah nihil soal pemberantasan covid. Gambling udah siapa kuat dia bertahan hidup yang lemah maka mati. Padahal dinegara lain begitu serius menangani virus tsb.'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lemma = WordNetLemmatizer()\n",
        "stemmer = PorterStemmer()\n",
        "stop_words = set(stopwords.words('indonesian','english'))                       #\n",
        "def Cleantweet(txt):\n",
        "  txt = re.sub(r'http\\s+', ' ', txt)\n",
        "  txt = re.sub('[^a-zA-Z]', ' ', txt)\n",
        "  txt = str(txt).lower()\n",
        "  txt = word_tokenize(txt)\n",
        "  txt = [item for item in txt if item not in stop_words]\n",
        "  txt = [stemmer.stem(i) for i in txt]\n",
        "  # txt = [lemma.lemmatize(word=w,pos='v') for w in txt]\n",
        "  txt = [i for i in txt if len(i) > 2]\n",
        "  txt = ' '.join(txt)\n",
        "  return txt\n",
        "data['Cleantweet'] = data['tweet'].apply(Cleantweet)"
      ],
      "metadata": {
        "id": "dVc_HJHWmgCR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 641
        },
        "id": "CnTKTrnEsII2",
        "outputId": "bd7dc527-1e0f-4e45-87be-56e101f5e4bb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                 tweet Sentimen  \\\n",
              "0     https://bit.ly/2L6CcbB  | Seharusnya saat ini...  Negatif   \n",
              "1    Klo kata gw Pemerintah tuh lagi menerapkan Her...  Negatif   \n",
              "2    Saat ini yang bisa saya lakukan hanya menyiapk...  Negatif   \n",
              "3    emosi bgt, lg kondisi begini gue disuruh liput...  Negatif   \n",
              "4    Bang gimna pemerintah mau peduli RRC urus ABK ...  Negatif   \n",
              "..                                                 ...      ...   \n",
              "995  Kalian tak percaya Covid ada, himbauan pemerin...  Positif   \n",
              "996  Berat perjuangan pemerintah Aceh, ketika rakya...  Positif   \n",
              "997  Anjuran pemerintah tidak boleh berpegangan tan...  Positif   \n",
              "998  gue bingung data covid naik terus, yang salah ...  Positif   \n",
              "999  Beneran kalo sakit atau penyakit gw kambuh seb...  Negatif   \n",
              "\n",
              "                                            Cleantweet  \n",
              "0    http bit ccbb pemerintah lampung bantuan mahas...  \n",
              "1    klo pemerintah tuh menerapkan herd immun kemam...  \n",
              "2    lakukan makanan berbuka sahur kepala keluarga ...  \n",
              "3    emosi bgt kondisi gue disuruh liputan covid do...  \n",
              "4    bang gimna pemerintah peduli rrc uru abk menin...  \n",
              "..                                                 ...  \n",
              "995  percaya covid himbauan pemerintah diabaikan br...  \n",
              "996  berat perjuangan pemerintah aceh rakyatnya per...  \n",
              "997  anjuran pemerintah berpegangan tangan mencegah...  \n",
              "998  gue bingung data covid salah pemerintah masyar...  \n",
              "999  beneran kalo sakit penyakit kambuh sebisa gama...  \n",
              "\n",
              "[1000 rows x 3 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e415c35c-7026-4e4a-a2ba-749d950a77ac\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
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
              "      <th>tweet</th>\n",
              "      <th>Sentimen</th>\n",
              "      <th>Cleantweet</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>https://bit.ly/2L6CcbB  | Seharusnya saat ini...</td>\n",
              "      <td>Negatif</td>\n",
              "      <td>http bit ccbb pemerintah lampung bantuan mahas...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Klo kata gw Pemerintah tuh lagi menerapkan Her...</td>\n",
              "      <td>Negatif</td>\n",
              "      <td>klo pemerintah tuh menerapkan herd immun kemam...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Saat ini yang bisa saya lakukan hanya menyiapk...</td>\n",
              "      <td>Negatif</td>\n",
              "      <td>lakukan makanan berbuka sahur kepala keluarga ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>emosi bgt, lg kondisi begini gue disuruh liput...</td>\n",
              "      <td>Negatif</td>\n",
              "      <td>emosi bgt kondisi gue disuruh liputan covid do...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Bang gimna pemerintah mau peduli RRC urus ABK ...</td>\n",
              "      <td>Negatif</td>\n",
              "      <td>bang gimna pemerintah peduli rrc uru abk menin...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>995</th>\n",
              "      <td>Kalian tak percaya Covid ada, himbauan pemerin...</td>\n",
              "      <td>Positif</td>\n",
              "      <td>percaya covid himbauan pemerintah diabaikan br...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>996</th>\n",
              "      <td>Berat perjuangan pemerintah Aceh, ketika rakya...</td>\n",
              "      <td>Positif</td>\n",
              "      <td>berat perjuangan pemerintah aceh rakyatnya per...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>997</th>\n",
              "      <td>Anjuran pemerintah tidak boleh berpegangan tan...</td>\n",
              "      <td>Positif</td>\n",
              "      <td>anjuran pemerintah berpegangan tangan mencegah...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>998</th>\n",
              "      <td>gue bingung data covid naik terus, yang salah ...</td>\n",
              "      <td>Positif</td>\n",
              "      <td>gue bingung data covid salah pemerintah masyar...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>999</th>\n",
              "      <td>Beneran kalo sakit atau penyakit gw kambuh seb...</td>\n",
              "      <td>Negatif</td>\n",
              "      <td>beneran kalo sakit penyakit kambuh sebisa gama...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>1000 rows × 3 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e415c35c-7026-4e4a-a2ba-749d950a77ac')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-e415c35c-7026-4e4a-a2ba-749d950a77ac button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-e415c35c-7026-4e4a-a2ba-749d950a77ac');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "Dw1Ccx2iR2Xg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Perform SVM"
      ],
      "metadata": {
        "id": "xh23U_dmtm6V"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# split x dan y\n",
        "x = data['Cleantweet']\n",
        "y = data['Sentimen']\n",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)"
      ],
      "metadata": {
        "id": "Tw8k6e-jtryC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_test"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MGVyxyq6vzW-",
        "outputId": "8edcaebb-1580-49f3-b33f-ceb9ac6249ee"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "695    pemerintah aceh berhasil intervensi aspek kehi...\n",
              "290    sembilan sepuluh pemerintah indonesia takkan m...\n",
              "512    puse sejalan pemerintah pusat selesainya covid...\n",
              "411    bingung pemerintah data covid terinfeksi posit...\n",
              "403    salam membara ditengah pandemik covid menyelen...\n",
              "                             ...                        \n",
              "379    viru covid berlanjut psbb terapkan daerah keme...\n",
              "756    penerapan psbb melulu penindakan keikitsertaan...\n",
              "943    pandemi covid diambang gari merah meruntuhkan ...\n",
              "368    mari kembalikan penanggulan wabah covid pemeri...\n",
              "820    maaf who kayaknya salah paham sedari dipersiap...\n",
              "Name: Cleantweet, Length: 200, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# perform count vectorizer\n",
        "vectorizer = CountVectorizer()\n",
        "vectorizer.fit(x_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bnBmT17zKJTu",
        "outputId": "41d19da7-af58-429a-f48c-1af7e0d04e6e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "CountVectorizer()"
            ]
          },
          "metadata": {},
          "execution_count": 38
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# x_train\n",
        "x_train = vectorizer.transform(x_train)\n",
        "x_test = vectorizer.transform(x_test)"
      ],
      "metadata": {
        "id": "WjlhzV1dMaq-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train.toarray()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SdxOFxRhM2Lf",
        "outputId": "5ffdf979-bf7b-4d1a-a93b-dd98993deacd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0, 0, 0, ..., 0, 0, 0],\n",
              "       [0, 0, 0, ..., 0, 0, 0],\n",
              "       [0, 0, 0, ..., 0, 0, 0],\n",
              "       ...,\n",
              "       [0, 0, 0, ..., 0, 0, 0],\n",
              "       [0, 0, 0, ..., 0, 0, 0],\n",
              "       [0, 0, 0, ..., 0, 0, 0]])"
            ]
          },
          "metadata": {},
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import svm\n",
        "#Create a svm Classifier\n",
        "clf = svm.SVC(kernel='linear') \n",
        "# Linear Kernel#Train the model using the training sets\n",
        "clf.fit(x_train, y_train) \n",
        "#Predict the response for test dataset\n",
        "y_pred = clf.predict(x_test)"
      ],
      "metadata": {
        "id": "dyaZZqf7SIk2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J6s-RDmvTN9m",
        "outputId": "04ef6edb-0b47-4b14-fd8e-0ea382a5adbd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['Negatif', 'Negatif', 'Negatif', 'Negatif', 'Positif', 'Negatif',\n",
              "       'Positif', 'Positif', 'Negatif', 'Positif', 'Positif', 'Negatif',\n",
              "       'Positif', 'Negatif', 'Negatif', 'Negatif', 'Negatif', 'Positif',\n",
              "       'Negatif', 'Negatif', 'Positif', 'Positif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Negatif', 'Negatif', 'Negatif', 'Positif', 'Positif',\n",
              "       'Negatif', 'Positif', 'Negatif', 'Positif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Positif', 'Negatif', 'Negatif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Positif', 'Negatif', 'Negatif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Positif', 'Positif', 'Negatif', 'Positif', 'Positif',\n",
              "       'Positif', 'Negatif', 'Negatif', 'Negatif', 'Positif', 'Negatif',\n",
              "       'Negatif', 'Positif', 'Negatif', 'Positif', 'Positif', 'Negatif',\n",
              "       'Positif', 'Positif', 'Negatif', 'Negatif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Negatif', 'Negatif', 'Positif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Negatif', 'Negatif', 'Negatif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Negatif', 'Positif', 'Positif', 'Negatif', 'Negatif',\n",
              "       'Positif', 'Negatif', 'Negatif', 'Positif', 'Positif', 'Negatif',\n",
              "       'Negatif', 'Negatif', 'Negatif', 'Negatif', 'Positif', 'Negatif',\n",
              "       'Negatif', 'Positif', 'Positif', 'Positif', 'Negatif', 'Positif',\n",
              "       'Positif', 'Positif', 'Negatif', 'Negatif', 'Positif', 'Negatif',\n",
              "       'Negatif', 'Positif', 'Positif', 'Negatif', 'Negatif', 'Negatif',\n",
              "       'Positif', 'Positif', 'Negatif', 'Negatif', 'Negatif', 'Negatif',\n",
              "       'Positif', 'Negatif', 'Negatif', 'Positif', 'Positif', 'Positif',\n",
              "       'Negatif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Positif',\n",
              "       'Positif', 'Negatif', 'Negatif', 'Positif', 'Positif', 'Positif',\n",
              "       'Positif', 'Negatif', 'Positif', 'Positif', 'Positif', 'Negatif',\n",
              "       'Positif', 'Negatif', 'Positif', 'Positif', 'Negatif', 'Positif',\n",
              "       'Negatif', 'Negatif', 'Positif', 'Positif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Negatif', 'Negatif', 'Positif', 'Negatif', 'Positif',\n",
              "       'Positif', 'Positif', 'Negatif', 'Negatif', 'Positif', 'Positif',\n",
              "       'Positif', 'Negatif', 'Negatif', 'Negatif', 'Positif', 'Positif',\n",
              "       'Positif', 'Negatif', 'Positif', 'Positif', 'Negatif', 'Negatif',\n",
              "       'Negatif', 'Negatif', 'Positif', 'Negatif', 'Negatif', 'Positif',\n",
              "       'Positif', 'Positif', 'Negatif', 'Negatif', 'Positif', 'Negatif',\n",
              "       'Negatif', 'Positif'], dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 46
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_test"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z4YGrppxTXSI",
        "outputId": "aa946b3e-e013-4ef2-92d9-992f842d0167"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "695    Positif\n",
              "290    Negatif\n",
              "512    Negatif\n",
              "411    Negatif\n",
              "403    Positif\n",
              "        ...   \n",
              "379    Negatif\n",
              "756    Positif\n",
              "943    Negatif\n",
              "368    Positif\n",
              "820    Negatif\n",
              "Name: Sentimen, Length: 200, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 47
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix\n",
        "confusion_matrix(y_test, y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TltS1sI0Te-G",
        "outputId": "bdfeba13-db91-4dfa-bbec-fb8e154de18b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[81, 16],\n",
              "       [36, 67]])"
            ]
          },
          "metadata": {},
          "execution_count": 48
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = clf.predict(x_test)\n",
        "print ('Accuracy of SVM classifier on test set: {:.2f}' .format(clf.score(x_test, y_test)))\n",
        "confusion_matrix = confusion_matrix(y_test, y_pred)\n",
        "print(confusion_matrix)\n",
        "print(classification_report(y_test, y_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NbhsepBEUz7X",
        "outputId": "75ed76eb-dc60-4144-994a-665e66d2a599"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy of SVM classifier on test set: 0.74\n",
            "[[81 16]\n",
            " [36 67]]\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "     Negatif       0.69      0.84      0.76        97\n",
            "     Positif       0.81      0.65      0.72       103\n",
            "\n",
            "    accuracy                           0.74       200\n",
            "   macro avg       0.75      0.74      0.74       200\n",
            "weighted avg       0.75      0.74      0.74       200\n",
            "\n"
          ]
        }
      ]
    }
  ]
}