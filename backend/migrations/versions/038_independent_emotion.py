"""Run facial-expression analysis as an independent, encrypted video job lane."""

from alembic import op
import sqlalchemy as sa


revision = "038_independent_emotion"
down_revision = "037_encrypted_video_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing rows keep their historical behaviour: any emotion result they
    # have stays inside the VSViG result payload and is still served from it.
    with op.batch_alter_table("video_detection_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "emotion_status",
                sa.String(length=24),
                nullable=False,
                server_default="not_applicable",
            )
        )
        batch_op.add_column(sa.Column("emotion_nonce", sa.LargeBinary(), nullable=True))
        batch_op.add_column(sa.Column("emotion_ciphertext", sa.LargeBinary(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("video_detection_jobs") as batch_op:
        batch_op.drop_column("emotion_ciphertext")
        batch_op.drop_column("emotion_nonce")
        batch_op.drop_column("emotion_status")
