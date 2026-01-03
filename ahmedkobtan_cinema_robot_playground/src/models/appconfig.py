from pydantic import BaseModel


class Resolved(BaseModel):
    env: str
    ip_webcam_url: str


class ConfigResolution(BaseModel):
    resolved: Resolved


class AppConfig(BaseModel):
    configResolution: ConfigResolution
    deliveryVersion: str


# class SurveyType(str, Enum):
#     quant = "quant"
#     qual = "qual"


# class UUIDProcesserInput(BaseModel):
#     survey_type: SurveyType = Field(
#         ..., description="The survey type can be either 'quant' or 'qual'"
#     )
#     target_completes: int = Field(
#         ..., description="The target number of completes", gt=0
#     )
#     project_id: str = Field(..., description="The project ID")
#     survey_id: str = Field(..., description="The survey ID")
#     batch_id: int = Field(..., description="The batch ID", gt=0)
#     n_qualified_so_far: int = Field(..., description="The number of qualified so far")
#     n_invited_so_far: int = Field(..., description="The number of invited so far")

#     panelist_ids_df: pd.DataFrame = Field(
#         ..., description="The panelist IDs dataframe read from UUID file path"
#     )
#     panelist_features_df: pd.DataFrame = Field(
#         ..., description="The panelist features dataframe"
#     )
#     qualified_complete_points: int = Field(
#         ...,
#         description="The maximum reward points a panelist can receive from this survey",
#     )
#     cnt_qtype_radios: int = Field(..., description="The number of radio questions")
#     cnt_qtype_checkboxes: int = Field(
#         ..., description="The number of checkbox questions"
#     )
#     cnt_qtype_matrix: int = Field(..., description="The number of matrix questions")
#     cnt_qtype_range: int = Field(..., description="The number of range questions")
#     cnt_qtype_ranking: int = Field(..., description="The number of ranking questions")
#     cnt_qtype_text: int = Field(..., description="The number of text questions")
#     cnt_qtype_media: int = Field(..., description="The number of media questions")
#     cnt_qtype_video: int = Field(..., description="The number of video questions")
#     cnt_qtype_select: int = Field(..., description="The number of select questions")
#     cnt_qtype_image: int = Field(..., description="The number of image questions")
#     cnt_q_termination: int = Field(
#         ..., description="The number of termination questions"
#     )

#     inferencer: inference_bounded_regressor = Field(
#         ..., description="The inferencer to predict response rate using trained model"
#     )

#     @model_validator(mode="before")
#     @classmethod
#     def validate_so_far(cls, values):
#         batch_id = values.get("batch_id")
#         n_qualified_so_far = values.get("n_qualified_so_far")
#         n_invited_so_far = values.get("n_invited_so_far")
#         qualified_complete_points = values.get("qualified_complete_points")
#         target_completes = values.get("target_completes")

#         if batch_id == 1:
#             if n_qualified_so_far != 0:
#                 raise ValueError("n_qualified_so_far must be 0 when batch_id is 1.")
#             if n_invited_so_far != 0:
#                 raise ValueError("n_invited_so_far must be 0 when batch_id is 1.")
#         elif batch_id > 1:
#             if n_qualified_so_far < 0:
#                 raise ValueError(
#                     "n_qualified_so_far must be greater than or equal to 0 when batch_id is greater than 1."
#                 )
#             if n_invited_so_far <= 0:
#                 raise ValueError(
#                     "n_invited_so_far must be greater than 0 when batch_id is greater than 1."
#                 )
#         if qualified_complete_points < 0:
#             raise ValueError(
#                 "qualified_complete_points must be greater than or equal to 0"
#             )
#         if n_qualified_so_far >= target_completes:
#             raise ValueError("target_completes must be greater than n_qualified_so_far")

#         return values

# class Config:
#     arbitrary_types_allowed = True
