"""Populate DB

Revision ID: f3233f1e7fa6
Revises: 3acc466475a7
Create Date: 2024-09-13 09:52:46.816872

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3233f1e7fa6'
down_revision: Union[str, None] = '3acc466475a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    service_table = sa.sql.table(
        'ml_service',
        sa.Column('name', sa.String(), primary_key=True),
        sa.Column('description', sa.String(), nullable=True),
    )
    service_data = [
        {
            "name": "echo",
            "description": "Echo model",
        },
        {
            "name": "catboost_prophet",
            "description": "Demand forecasting model",
        },
    ]
    op.bulk_insert(service_table, service_data)


def downgrade() -> None:
    op.execute('DELETE FROM ml_service')
